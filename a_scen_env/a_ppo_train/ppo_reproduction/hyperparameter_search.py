#!/usr/bin/env python3
"""
PPO Expert复现超参数搜索工具
用于批量实验和超参数优化
"""

import os
import sys
import json
import subprocess
import time
import itertools
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd


def _worker_run(script_path_str: str, args_dict_local: Dict, exp_suffix: str, save_dir_hint: str):
    """子进程执行器：用于多进程并行训练"""
    # 导入需要的模块
    import subprocess
    import time
    from datetime import datetime
    from pathlib import Path
    import pandas as pd
    
    start_time = time.time()
    cmd = ["python", script_path_str]
    for k, v in args_dict_local.items():
        cmd.extend([f"--{k}", str(v)])

    # 每个 trial 用自己的目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    custom_save_dir = Path(save_dir_hint) / "hypersearch_grid" / f"hypersearch_grid_{exp_suffix}_{timestamp}"
    # 修复：把 save_dir 传"自己的目录"而非父目录
    cmd.extend(["--save_dir", str(custom_save_dir)])

    # try:
    print(f"🚀 启动训练: {' '.join(cmd[-12:])}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if result.returncode != 0:
        return {
            "args": args_dict_local,
            "status": "failed",
            "error": result.stderr,
            "duration": time.time() - start_time,
            "exp_dir": str(custom_save_dir)
        }

    # 尝试读取结果
    out = {
        "args": args_dict_local,
        "status": "completed",
        "duration": time.time() - start_time,
        "exp_dir": str(custom_save_dir)
    }
    try:
        # 查找CSV文件，考虑嵌套的runs目录结构
        csv_paths = [
            custom_save_dir / "training_logs.csv",  # 直接路径
            custom_save_dir / "runs" / "ppo_expert_reproduction*" / "training_logs.csv"  # 嵌套路径
        ]
        
        csv_path = None
        for path_pattern in csv_paths:
            if "*" in str(path_pattern):
                # 使用glob查找匹配的路径
                import glob
                matches = glob.glob(str(path_pattern))
                if matches:
                    csv_path = Path(matches[0])
                    break
            elif path_pattern.exists():
                csv_path = path_pattern
                break
        
        if csv_path and csv_path.exists():
            df = pd.read_csv(csv_path)
            if not df.empty:
                out["best_reward"] = df["ep_reward_mean"].max()
                out["final_reward"] = df["ep_reward_mean"].iloc[-1]
                if "collision_rate" in df.columns:
                    out["final_collision_rate"] = df["collision_rate"].iloc[-1]
                if "success_rate" in df.columns:
                    out["final_success_rate"] = df["success_rate"].iloc[-1]
                out["training_steps"] = df["step"].iloc[-1]
                out["csv_found"] = str(csv_path)  # 调试信息
            else:
                out["warn_extract"] = "CSV文件为空"
        else:
            out["warn_extract"] = f"未找到CSV文件，搜索路径: {[str(p) for p in csv_paths]}"
    except Exception as e:
        out["warn_extract"] = str(e)

    return out

    # except subprocess.TimeoutExpired:
    #     return {
    #         "args": args_dict_local,
    #         "status": "timeout",
    #         "duration": time.time() - start_time,
    #         "exp_dir": str(custom_save_dir)
    #     }
    # except Exception as e:
    #     return {
    #         "args": args_dict_local,
    #         "status": "error",
    #         "error": str(e),
    #         "duration": time.time() - start_time,
    #         "exp_dir": str(custom_save_dir)
    #     }


class HyperparameterSearch:
    """超参数搜索器"""
    
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.script_path = self.base_dir / "ppo_expert_reproduction.py"
        self.results = []
        
        # 确保脚本存在
        if not self.script_path.exists():
            raise FileNotFoundError(f"训练脚本不存在: {self.script_path}")
    
    # def grid_search(self, param_grid: Dict[str, List], base_args: Dict = None, max_concurrent: int = 2):
    #     """网格搜索"""
    #     print(f"🔍 开始网格搜索超参数优化")
    #     print(f"📊 参数网格: {param_grid}")
        
    #     # 生成参数组合
    #     param_names = list(param_grid.keys())
    #     param_values = list(param_grid.values())
    #     param_combinations = list(itertools.product(*param_values))
        
    #     print(f"🎯 总共 {len(param_combinations)} 种参数组合")
        
    #     # 基础参数
    #     if base_args is None:
    #         base_args = {
    #             "total_timesteps": 500000,  # 减少训练时间用于搜索
    #             "eval_freq": 5,
    #             "checkpoint_freq": 25,
    #             "log_freq": 1
    #         }
        
    #     # 执行实验
    #     for i, combination in enumerate(param_combinations):
    #         print(f"\n{'='*60}")
    #         print(f"实验 {i+1}/{len(param_combinations)}")
            
    #         # 构建参数字典
    #         experiment_args = base_args.copy()
    #         for param_name, value in zip(param_names, combination):
    #             experiment_args[param_name] = value
            
    #         print(f"参数组合: {dict(zip(param_names, combination))}")
            
    #         # 运行实验
    #         result = self._run_single_experiment(experiment_args, f"grid_search_{i+1:03d}")
    #         self.results.append(result)
            
    #         # 显示当前最佳结果
    #         self._show_current_best()

    def grid_search(self, param_grid: Dict[str, List], base_args: Dict = None, max_concurrent: int = 1):
        """网格搜索（并行版）"""
        print(f"🔍 开始网格搜索超参数优化")
        print(f"📊 参数网格: {param_grid}")

        # 生成参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        print(f"🎯 总共 {len(param_combinations)} 种参数组合")

        # 基础参数（快速评估）
        # if base_args is None:
        base_args = {
            "total_timesteps": 10,
            "eval_freq": 5,
            "checkpoint_freq": 25,
            "log_freq": 1
        }

        # 先做一次性过滤：只提交满足整除约束的组合
        filtered_jobs = []
        for combo in param_combinations:
            args_dict = self._merge_args(base_args, param_names, combo)
            if self._is_valid_combo(args_dict):
                filtered_jobs.append((combo, args_dict))
            else:
                print(f"⚠️ 跳过不满足整除约束的组合: {dict(zip(param_names, combo))} "
                    f"(需要 (n_steps*n_envs)%batch_size==0)")

        print(f"✅ 将执行 {len(filtered_jobs)} 个有效组合（并行度 {max_concurrent}）")

        # 提交并行任务
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # 使用模块级的 _worker_run 函数，避免局部函数的pickle序列化问题

        # 真正并行跑
        futures = []
        save_root = str(self.base_dir / "runs")
        with ProcessPoolExecutor(max_workers=max_concurrent) as ex:
            for idx, (combo, args_dict) in enumerate(filtered_jobs, start=1):
                suffix = f"{idx:03d}_" + "_".join(f"{k}-{v}" for k, v in zip(param_names, combo))
                futures.append(
                    ex.submit(_worker_run, str(self.script_path), args_dict, suffix, save_root)
                )

            # 回收结果
            for fut in as_completed(futures):
                res = fut.result()
                self.results.append(res)

                # 打印即时最优
                successful = [r for r in self.results if r.get("status") == "completed" and "best_reward" in r]
                # if successful:
                best = max(successful, key=lambda x: x["best_reward"])
                print(f"\n🏆 当前最佳: best_reward={best['best_reward']:.3f} | "
                    f"{self._format_key_params(best['args'])}")
                # else:
                #     print("⏳ 仍无成功完成的实验")
    
    def random_search(self, param_distributions: Dict[str, Tuple], n_trials: int = 10, 
                     base_args: Dict = None):
        """随机搜索"""
        print(f"🎲 开始随机搜索超参数优化")
        print(f"🎯 试验次数: {n_trials}")
        
        import random
        
        # 基础参数
        if base_args is None:
            base_args = {
                "total_timesteps": 300000,  # 更少步数用于快速搜索
                "eval_freq": 5,
                "checkpoint_freq": 25
            }
        
        for trial in range(n_trials):
            print(f"\n{'='*60}")
            print(f"随机试验 {trial+1}/{n_trials}")
            
            # 随机采样参数
            experiment_args = base_args.copy()
            sampled_params = {}
            
            for param_name, (dist_type, *dist_args) in param_distributions.items():
                if dist_type == "uniform":
                    low, high = dist_args
                    value = random.uniform(low, high)
                elif dist_type == "log_uniform":
                    low, high = dist_args
                    value = 10 ** random.uniform(np.log10(low), np.log10(high))
                elif dist_type == "choice":
                    choices = dist_args[0]
                    value = random.choice(choices)
                elif dist_type == "int_uniform":
                    low, high = dist_args
                    value = random.randint(low, high)
                else:
                    raise ValueError(f"未知分布类型: {dist_type}")
                
                experiment_args[param_name] = value
                sampled_params[param_name] = value
            
            print(f"采样参数: {sampled_params}")
            
            # 运行实验
            result = self._run_single_experiment(experiment_args, f"random_search_{trial+1:03d}")
            self.results.append(result)
            
            # 显示当前最佳结果
            self._show_current_best()
    
    def bayesian_search(self, param_space: Dict, n_trials: int = 15):
        """贝叶斯优化搜索 (需要optuna)"""
        try:
            import optuna
        except ImportError:
            print("❌ 贝叶斯搜索需要optuna库: pip install optuna")
            return
        
        print(f"🧠 开始贝叶斯优化搜索")
        print(f"🎯 试验次数: {n_trials}")
        
        def objective(trial):
            # 根据参数空间采样参数
            experiment_args = {
                "total_timesteps": 200000,  # 快速评估
                "eval_freq": 5,
                "checkpoint_freq": 25
            }
            
            sampled_params = {}
            for param_name, param_config in param_space.items():
                if param_config["type"] == "float":
                    value = trial.suggest_float(
                        param_name, 
                        param_config["low"], 
                        param_config["high"],
                        log=param_config.get("log", False)
                    )
                elif param_config["type"] == "int":
                    value = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"]
                    )
                elif param_config["type"] == "categorical":
                    value = trial.suggest_categorical(
                        param_name,
                        param_config["choices"]
                    )
                else:
                    raise ValueError(f"未知参数类型: {param_config['type']}")
                
                experiment_args[param_name] = value
                sampled_params[param_name] = value
            
            print(f"试验 {trial.number + 1}: {sampled_params}")
            
            # 运行实验
            result = self._run_single_experiment(
                experiment_args, 
                f"bayesian_{trial.number + 1:03d}"
            )
            
            self.results.append(result)
            
            # 返回优化目标 (负数因为optuna默认最小化)
            return -result.get("best_reward", -float('inf'))
        
        # 创建优化器
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        print(f"\n🏆 贝叶斯搜索完成!")
        print(f"最佳参数: {study.best_params}")
        print(f"最佳目标值: {-study.best_value:.3f}")
    

    def _is_valid_combo(self, args: Dict) -> bool:
        """SB3 约束：总 batch (= n_steps * n_envs) 必须被 batch_size 整除"""
        try:
            n_steps = int(args.get("n_steps", 0))
            n_envs = int(args.get("n_envs", 0))
            batch_size = int(args.get("batch_size", 0))
            if n_steps > 0 and n_envs > 0 and batch_size > 0:
                return (n_steps * n_envs) % batch_size == 0
            # 没给齐三个参数就不做约束
            return True
        except Exception:
            return True


    def _merge_args(self, base_args: Dict, param_names: List[str], values: Tuple) -> Dict:
        """把一组参数组合并进 base_args，返回新的 dict"""
        merged = dict(base_args or {})
        for k, v in zip(param_names, values):
            merged[k] = v
        return merged


    def _run_single_experiment(self, args_dict: Dict, exp_suffix: str) -> Dict:
        """运行单个实验"""
        start_time = time.time()
        
        # 构建命令行参数
        cmd = ["python", str(self.script_path)]
        
        for key, value in args_dict.items():
            cmd.extend([f"--{key}", str(value)])
        
        # 添加实验后缀到保存目录参数中
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        custom_save_dir = self.base_dir / "runs" / f"hypersearch_{exp_suffix}_{timestamp}"
        cmd.extend(["--save_dir", str(custom_save_dir)])
        
        try:
            # 运行训练
            print(f"🚀 启动训练: {' '.join(cmd[-10:])}")  # 显示最后几个参数
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2小时超时
            
            if result.returncode != 0:
                print(f"❌ 训练失败: {result.stderr}")
                return {
                    "args": args_dict,
                    "status": "failed",
                    "error": result.stderr,
                    "duration": time.time() - start_time
                }
            
            # 读取结果
            exp_result = self._extract_experiment_result(custom_save_dir)
            exp_result.update({
                "args": args_dict,
                "status": "completed",
                "duration": time.time() - start_time,
                "exp_dir": str(custom_save_dir)
            })
            
            print(f"✅ 训练完成，最佳奖励: {exp_result.get('best_reward', 'N/A'):.3f}")
            return exp_result
            
        except subprocess.TimeoutExpired:
            print(f"⏰ 训练超时")
            return {
                "args": args_dict,
                "status": "timeout", 
                "duration": time.time() - start_time
            }
        except Exception as e:
            print(f"❌ 实验失败: {e}")
            return {
                "args": args_dict,
                "status": "error",
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    def _extract_experiment_result(self, exp_dir: Path) -> Dict:
        """从实验目录提取结果"""
        result = {}
        
        try:
            # 从训练日志CSV提取结果
            csv_path = exp_dir / "training_logs.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                if not df.empty:
                    result["best_reward"] = df["ep_reward_mean"].max()
                    result["final_reward"] = df["ep_reward_mean"].iloc[-1]
                    result["final_collision_rate"] = df["collision_rate"].iloc[-1]
                    result["final_success_rate"] = df["success_rate"].iloc[-1]
                    result["training_steps"] = df["step"].iloc[-1]
            
            # 从配置文件提取超参数
            config_path = exp_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    result["hyperparameters"] = config.get("hyperparameters", {})
        
        except Exception as e:
            print(f"⚠️ 结果提取失败: {e}")
        
        return result
    
    def _show_current_best(self):
        """显示当前最佳结果"""
        if not self.results:
            return
        
        # 过滤成功的实验
        successful_results = [r for r in self.results if r.get("status") == "completed" and "best_reward" in r]
        
        if not successful_results:
            print("⚠️ 暂无成功完成的实验")
            return
        
        # 找到最佳结果
        best_result = max(successful_results, key=lambda x: x["best_reward"])
        
        print(f"\n🏆 当前最佳结果:")
        print(f"   最佳奖励: {best_result['best_reward']:.3f}")
        print(f"   最终奖励: {best_result.get('final_reward', 'N/A'):.3f}")
        print(f"   碰撞率: {best_result.get('final_collision_rate', 'N/A'):.3f}")
        print(f"   成功率: {best_result.get('final_success_rate', 'N/A'):.3f}")
        print(f"   关键参数: {self._format_key_params(best_result['args'])}")
    
    def _format_key_params(self, args: Dict) -> str:
        """格式化关键参数显示"""
        key_params = ["lr", "n_steps", "n_envs", "batch_size", "clip_range", "entropy_coef"]
        formatted = []
        for param in key_params:
            if param in args:
                formatted.append(f"{param}={args[param]}")
        return ", ".join(formatted)
    
    def generate_search_report(self, output_path: str = None):
        """生成搜索报告"""
        if output_path is None:
            output_path = self.base_dir / "hyperparameter_search_report.md"
        
        # 过滤成功的实验
        successful_results = [r for r in self.results if r.get("status") == "completed" and "best_reward" in r]
        
        if not successful_results:
            print("⚠️ 没有成功的实验可生成报告")
            return
        
        # 按最佳奖励排序
        successful_results.sort(key=lambda x: x["best_reward"], reverse=True)
        
        # 生成报告
        report_content = f"""# PPO Expert超参数搜索报告

## 📊 搜索概览

- **总实验数**: {len(self.results)}
- **成功实验数**: {len(successful_results)}
- **失败实验数**: {len(self.results) - len(successful_results)}
- **搜索时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🏆 最佳结果 Top 5

"""
        
        for i, result in enumerate(successful_results[:5]):
            report_content += f"""### 第 {i+1} 名
- **最佳奖励**: {result['best_reward']:.3f}
- **最终奖励**: {result.get('final_reward', 'N/A'):.3f}
- **碰撞率**: {result.get('final_collision_rate', 'N/A'):.3f}
- **成功率**: {result.get('final_success_rate', 'N/A'):.3f}
- **训练时长**: {result.get('duration', 0)/3600:.1f} 小时
- **关键参数**:
"""
            args = result['args']
            for param in ['lr', 'n_steps', 'n_envs', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'clip_range', 'entropy_coef']:
                if param in args:
                    report_content += f"  - {param}: {args[param]}\n"
            
            report_content += "\n"
        
        # 参数分析
        report_content += """## 📈 参数影响分析

"""
        
        # 分析各参数与性能的关系
        df = pd.DataFrame([{**r['args'], 'best_reward': r['best_reward']} for r in successful_results])
        
        key_params = ['lr', 'n_steps', 'n_envs', 'batch_size', 'clip_range', 'entropy_coef']
        for param in key_params:
            if param in df.columns:
                correlation = df[param].corr(df['best_reward'])
                report_content += f"- **{param}**: 与性能相关性 {correlation:.3f}\n"
        
        report_content += f"""

## 💡 推荐配置

基于搜索结果，推荐以下配置用于正式训练：

```bash
./train.sh custom \\
"""
        
        # 使用最佳结果的参数
        best_args = successful_results[0]['args']
        for param in ['lr', 'n_steps', 'n_envs', 'batch_size', 'n_epochs', 'gamma', 'gae_lambda', 'clip_range', 'entropy_coef']:
            if param in best_args:
                report_content += f"    --{param} {best_args[param]} \\\n"
        
        report_content += f"""    --total_timesteps 2000000
```

## 📊 详细结果表

| 排名 | 最佳奖励 | 学习率 | Steps | Envs | Batch | Clip | 训练时长(h) |
|------|----------|--------|-------|------|-------|------|-------------|
"""
        
        for i, result in enumerate(successful_results[:10]):
            args = result['args']
            report_content += f"| {i+1} | {result['best_reward']:.3f} | "
            report_content += f"{args.get('lr', 'N/A')} | "
            report_content += f"{args.get('n_steps', 'N/A')} | "
            report_content += f"{args.get('n_envs', 'N/A')} | "
            report_content += f"{args.get('batch_size', 'N/A')} | "
            report_content += f"{args.get('clip_range', 'N/A')} | "
            report_content += f"{result.get('duration', 0)/3600:.1f} |\n"
        
        report_content += f"""

---
**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # 保存报告
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"📄 搜索报告已生成: {output_path}")
        
        # 同时保存CSV详细数据
        csv_path = str(output_path).replace('.md', '_details.csv')
        if successful_results:
            df_full = pd.DataFrame([{**r['args'], **{k:v for k,v in r.items() if k != 'args'}} for r in successful_results])
            df_full.to_csv(csv_path, index=False)
            print(f"📊 详细数据已保存: {csv_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="PPO Expert超参数搜索")
    parser.add_argument("--search_type", type=str, default="grid",
                       choices=["grid", "random", "bayesian"],
                       help="搜索类型")
    parser.add_argument("--n_trials", type=int, default=10,
                       help="试验次数 (random/bayesian)")
    parser.add_argument("--base_dir", type=str, 
                       default="/home/jxy/桌面/1_Project/20250705_computational_cognitive_modeling/computational_cognitive_modeling/metadrive/a_scen_env/a_ppo_train/ppo_reproduction",
                       help="基础目录")
    
    args = parser.parse_args()
    
    # 创建搜索器
    searcher = HyperparameterSearch(args.base_dir)
    
    print("🔍 PPO Expert超参数搜索工具")
    print("=" * 50)
    print(f"搜索类型: {args.search_type}")
    print(f"基础目录: {args.base_dir}")
    
    try:
        if args.search_type == "grid":
            # 网格搜索参数
            param_grid = {
                "lr": [1e-4, 3e-4, 1e-3],
                # "n_steps": [1024, 2048, 4096],
                # "n_envs": [4, 8],              # 增加搜索维度
                # "batch_size": [256, 512],      # 增加搜索维度
                # "clip_range": [0.1, 0.2, 0.3],
                # "entropy_coef": [0.005, 0.01, 0.02]
            }
            searcher.grid_search(param_grid)
            
        elif args.search_type == "random":
            # 随机搜索参数分布
            param_distributions = {
                "lr": ("log_uniform", 1e-5, 1e-3),
                "n_steps": ("choice", [1024, 2048, 4096]),
                "n_envs": ("int_uniform", 4, 16),
                "batch_size": ("choice", [128, 256, 512]),
                "clip_range": ("uniform", 0.1, 0.3),
                "entropy_coef": ("log_uniform", 0.001, 0.05)
            }
            searcher.random_search(param_distributions, args.n_trials)
            
        elif args.search_type == "bayesian":
            # 贝叶斯优化参数空间
            param_space = {
                "lr": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
                "n_steps": {"type": "categorical", "choices": [1024, 2048, 4096]},
                "n_envs": {"type": "int", "low": 4, "high": 16},
                "batch_size": {"type": "categorical", "choices": [128, 256, 512]},
                "clip_range": {"type": "float", "low": 0.1, "high": 0.3},
                "entropy_coef": {"type": "float", "low": 0.001, "high": 0.05, "log": True}
            }
            searcher.bayesian_search(param_space, args.n_trials)
        
        # 生成搜索报告
        searcher.generate_search_report()
        
        print("\n🎉 超参数搜索完成！")
        
    except KeyboardInterrupt:
        print("\n⚠️ 搜索被用户中断")
        if searcher.results:
            searcher.generate_search_report()
    except Exception as e:
        print(f"\n❌ 搜索失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # 添加numpy import for random search
    import numpy as np
    main() 