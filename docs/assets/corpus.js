var title_corpus = [
    ['MetaDrive: An Open-ended Driving Simulator <br>with Infinite Scenes from Procedural Generation', 'MetaDrive：一个拥有无限场景的开放式驾驶平台'],
    ['Overview of MetaDrive Simulator', 'MetaDrive仿真器的总览'],
    ['Procedural Generation of Driving Scenes', '驾驶场景的过程生成'],
    ['Result of Improved Generalization', '提升范化性的结果'],
]
var author_corpus = [
    ['Quanyi Li', '黎权毅'],
    ['Zhenghao Peng', '彭正皓'],
    ['Qihang Zhang', '章启航'],
    ['Cong Qiu', '邱聪'],
    ['Chunxiao Liu', '刘春晓'],
    ['Bolei Zhou', '周博磊'],
]
var institution_corpus = [
    ['<div><sup>1</sup>The Chinese University of Hong Kong, <sup>2</sup>SenseTime Research, <sup>3</sup>Zhejiang University</div>', 
    '<div><sup>1</sup>香港中文大学, <sup>2</sup>商汤科技, <sup>3</sup>浙江大学</div>'],
]
var text_corpus = [
    ['To better evaluate and improve the generalization of learning-based driving systems, we introduce an\
    open-ended\
    and highly configurable driving simulator called MetaDrive.\
    MetaDrive can generate a diverse set of driving scenes through procedural generalization from basic traffic building blocks.\
     Currently the simulator is used to study the generalization of the driving agents trained from reinforcement learning. See <a href="">paper</a> for more detail.\ ', 
    '为了更好地评估和改善基于强化学习的自动驾驶系统的泛化性能，我们提出了一个开放式、极易配置的模拟器：MetaDrive！MetaDrive可以通过过程生成技术生成无限多的地图。\
    目前我们利用这个仿真器来研究强化学习系统得到的智能体的范化性。欢迎访问 <a href="https://github.com/decisionforce/metadrive">github.com/decisionforce/metadrive</a> 来亲身感受！'],
    ['We first define the elementary road blocks as follows,','我们首先定义了如下的基础路块：'],
    ['we then follow the proposed algorithm of procedural generation to synthesize maps:', '随后我们用了过程生成的技术来生成地图：'],
    ['We exhibit more generated maps as follows, which are further turned into interactive environments for reinforcement learning of end-to-end driving.', '我们展示了更多生成的地图，它们都可以被转换成强化学习算法可以与之交互的环境：'],
    ['We show that when trained with more procedurally generated maps, the driving agents from reinofrcement learning have better generalization performance on unseen test maps, and can handle more complex scenarios. The detailed experimental results are in the paper. You can reproduce the experiment through <a href="https://github.com/decisionforce/metadrive-generalization-paper">our generalization experiment code</a>.', '在训练中见识过更多地图的智能体展现出了更优越的测试性能，可以应对更加复杂的场景。这说明了我们的MetaDrive赋予了智能体更强大的泛化能力！你可以参考<a href="https://github.com/decisionforce/metadrive-generalization-paper">我们范化实验的代码</a>来复现展示的结果。'],
    ['The demo video of the generalizable agent is shown as follows. You can run the agent on your local machine through the provided example in <a href="https://github.com/decisionforce/metadrive">the simulator codebase</a>.',
     '以下视频展示了我们的范化性智能体。您可以在自己的机器上通过<a href="https://github.com/decisionforce/metadrive">模拟器代码</a>中提供的样例体验该智能体。'],
    ['Download the vedio.','点击下载视频。'],
    ['Citation', '引用']
]
var bar_corpus = [
    ['<b>Code</b>', '<b>代码</b>'],
    ['<b>Documentation</b>', '<b>文档</b>'],
    ['<b>Paper</b>', '<b>论文</b>']
]
var vedio_corpus = [
    // ['<iframe src="https://www.youtube.com/embed/T368RveOY9g" frameborder=0\
    // style="position: absolute; top: 2.5%; left: 2.5%; width: 95%; height: 100%;"\
    // allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"\
    // allowfullscreen></iframe>',
    // '<iframe src="https://player.bilibili.com/player.html?aid=501174481&bvid=BV1pK411u7qw&cid=284390025&page=1" frameborder=0\
    // style="position: absolute; top: 2.5%; left: 2.5%; width: 95%; height: 100%;"\
    // allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"\
    // allowfullscreen>\
    //  </iframe>' ],
     ['<iframe src="https://www.youtube.com/embed/2nb8Mhriq0I" frameborder=0\
    style="position: absolute; top: 2.5%; left: 2.5%; width: 95%; height: 100%;"\
    allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"\
    allowfullscreen></iframe>', 
     '<iframe src="https://player.bilibili.com/player.html?aid=373258795&bvid=BV15Z4y137cz&cid=271979816&page=1" frameborder=0\
     style="position: absolute; top: 2.5%; left: 2.5%; width: 95%; height: 100%;"\
     allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture"\
     allowfullscreen>\
      </iframe>']
]


var lang_flag = 1;

$(document).ready(function(){
    $(".switch").click(function(){
        i = 0;
        lang_flag=this.id;
        $(".title").each(
            function(){
                $(this).html(title_corpus[i][lang_flag]);
                i=i+1;
            }
        );

        i=0
        $(".author a").each(
            function(){
                $(this).html(author_corpus[i][lang_flag]);
                i=i+1;
            }
        );

        i=0
        $(".institution").each(
            function(){
                $(this).html(institution_corpus[i][lang_flag]);
                i=i+1;
            }
        );

        i=0
        $(".text").each(
            function(){
                $(this).html(text_corpus[i][lang_flag]);
                i=i+1;
            }
        );

        i=0
        $(".bar").each(
            function(){
                $(this).html(bar_corpus[i][lang_flag]);
                i=i+1;
            }
        );

        i=0
        $(".vedio").each(
            function(){
                $(this).html(vedio_corpus[i][lang_flag]);
                i=i+1;
            }
        );

        lang_flag = 1-lang_flag;
    });
});