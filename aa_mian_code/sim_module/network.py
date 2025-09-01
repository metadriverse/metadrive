
import torch
import torch.nn as nn
import matplotlib


class PPONetwork(nn.Module):
    """PPO网络结构 - 支持动态观测维度"""
    
    def __init__(self, obs_dim: int = 275, action_dim: int = 2, hidden_dim: int = 256, use_cognitive_modules: bool = False):
        super(PPONetwork, self).__init__()
        
        # 强制根据认知模块启用状态确定观测维度
        if use_cognitive_modules:
            # 认知模块启用时，强制使用283维度（275原始 + 4认知参数 + 4mask）
            self.obs_dim = 283
            print(f"🧠 认知模块启用，强制使用283维度")
        else:
            # 认知模块不启用时，强制使用275维度
            self.obs_dim = 275
            print(f"🔧 认知模块禁用，强制使用275维度")
        
        # Actor网络
        self.actor_fc1 = nn.Linear(self.obs_dim, hidden_dim)
        self.actor_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor_out = nn.Linear(hidden_dim, action_dim * 2)  # mean + log_std
        
        # Critic网络
        self.critic_fc1 = nn.Linear(self.obs_dim, hidden_dim)
        self.critic_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_out = nn.Linear(hidden_dim, 1)
        
        # 激活函数
        self.tanh = nn.Tanh()
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs):
        """前向传播"""
        # 检查观测维度匹配
        if obs.shape[-1] != self.obs_dim:
            raise ValueError(f"观测维度不匹配: 期望{self.obs_dim}, 实际{obs.shape[-1]}")
        
        # Actor前向
        x_actor = self.tanh(self.actor_fc1(obs))
        x_actor = self.tanh(self.actor_fc2(x_actor))
        action_logits = self.actor_out(x_actor)
        
        # Critic前向
        x_critic = self.tanh(self.critic_fc1(obs))
        x_critic = self.tanh(self.critic_fc2(x_critic))
        value = self.critic_out(x_critic)
        
        return action_logits, value
    
    def get_action_and_value(self, obs, action=None, deterministic=False):
        """获取动作和价值"""
        action_logits, value = self.forward(obs)
        
        # 分离均值和标准差
        action_mean, action_log_std = torch.chunk(action_logits, 2, dim=-1)
        action_std = torch.exp(action_log_std)
        
        # 创建分布
        dist = torch.distributions.Normal(action_mean, action_std)
        
        if action is None:
            if deterministic:
                action = action_mean
            else:
                action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy, value.squeeze(-1)
