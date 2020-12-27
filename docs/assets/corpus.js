var title_corpus = [
    ['PGDrive: An Open-ended Driving Simulator <br>with Infinite Scenes', 'PGDrive：一个拥有无限场景的开放式驾驶平台'],
    ['Overview', '总览'],
    ['Results', '结果'],
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
    ['<sup>1</sup>The Chinese University of Hong Kong', '<sup>1</sup>香港中文大学'],
    ['<sup>2</sup>Sense Time Inc.', '<sup>2</sup>商汤科技'],
    ['<sup>3</sup>Zhejiang University', '<sup>3</sup>浙江大学'],
]
var text_corpus = [
    ['To better evaluate and improve the generalization of learning-based driving system, we introduce an open-ended\
    and highly configurable driving simulator called PGDrive.\
    PGDrive can generate infinite driving scenes through procedural generalization, which can benifit the research\
    on the generalization of learning system.\
    Please visit <a href="https://github.com/decisionforce/pgdrive">github.com/decisionforce/pgdrive</a> to enjoy\
    PGDrive!', '为了更好地评估和改善基于强化学习的自动驾驶系统的泛化性能，我们提出了一个开放式、极易配置的模拟器：PGDrive！PGDrive可以通过过程生成技术生成无限多的地图\
    ，从而助力学习系统泛化性能的研究。欢迎访问 <a href="https://github.com/decisionforce/pgdrive">github.com/decisionforce/pgdrive</a> 来亲身感受！'],
    ['We generate maps with these elementary road blocks:', '我们用了以下基础路块来生成地图：'],
    ['Here are various generated maps:', '以下是生成的地图：'],
    ['Agents trained with more maps show better test performance, showing that PGDrive endows agent with generality:', '在训练中见识过更多地图的智能体展现出了更优越的测试性能。这说明了我们的PGDrive赋予了智能体更强大的泛化能力！'],
    ['The training code for the paper can be founded<a href="https://github.com/decisionforce/pgdrive-generalization-paper">here</a>.', '我们的训练代码在<a href="https://github.com/decisionforce/pgdrive-generalization-paper">这里</a>。'],
    ['View our trained agents in the following video:', '请在如下视频中欣赏我们训练的智能体：'],
    ['Citation', '引用']
]
var bar_corpus = [
    ['Website', '网页'],
    ['Github Repo', 'GitHub仓库'],
    ['Documentation', '文档'],
    ['Paper', '论文'],
    ['切换中文', 'To English']
]
var lang_flag = 1;

$(document).ready(function(){
    $("button").click(function(){
        i = 0;
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
        $(".institution div").each(
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

        lang_flag = 1-lang_flag;
    });
});