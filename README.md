本项目致力于整合PyTorch框架下各文本分类方法。  
目前主要专注于单卡中文分类。

从命令行运行main.py文件实现实验目标。
具体的使用文档和更新日志，如果真的有人用的话我再写。我现在就主要关注“集成”，对“易用”的要求以后再满足。

# 1. 数据
目前的处理方式是直接将整个数据集加载到内存中，如果实在太大的话再优化。

目前可用的处理方式（通过`-dt`参数传入处理方式名称，`-df`传入原始数据集的文件夹路径）：
- `iflytek`：train.json和dev.json每一行是一个样本，可以用json包加载，sentence的值是文本，label的值是标签索引，label_des的值是标签文本；test.json的sentence值是文本，id值是数值；labels.json每行是一个标签，label的值是索引，label_des的值是文本，标签总数从该文件中获取
- `ChnSentiCorp_htl_all`：ChnSentiCorp_htl_all.csv是一个用逗号作为分隔符的CSV文件，label字段是标签（0-1），review字段是文本

本项目中，部分数据集原本就自带划分数据集指标。需要通过代码手动划分数据集的情况下，在上述处理方式名称后直接按顺序添加：
- random包随机种子
- 数据划分比例（例：`7-2-1`）

（这一部分需要考虑的情况还蛮多的，我以后慢慢补充吧）

理论上倒是也可以自动计算，但是以后再写吧，所以现在需要显式传入的参数：
`-od`：标签总数
## 1.1 中文二分类
- ChnSentiCorp_htl_all.csv：CSV文件下载源<https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv>，处理方式为`ChnSentiCorp_htl_all`
## 1.2 中文multi-class单标签分类
## 1.3 中文multi-class multi-label分类
- iflytek_public：压缩文件下载源<https://storage.googleapis.com/cluebenchmark/tasks/iflytek_public.zip>，处理方式为`iflytek`
## 1.4 中文多任务multi-class单标签分类

# 2. 表征方法
通过`-e`参数传入方法名称
- 通过word2vec模型得到词表征后，直接离线得到样本表征：天然与后面模型部分解耦（本项目直接将文本嵌入部分前置了。实际上甚至可以预先储存文本嵌入，或者模型一边运行文本一边嵌入，这些我以后再写）。需要分词
    - `w2v_mean`：使用word2vec向量在样本上的平均值，作为样本表征
    - 预训练词向量的路径通过`-ep`参数传入
    - 加载预训练词向量的方法通过`-et`参数传入：
        - `Chinese-Word-Vectors`：<https://github.com/Embedding/Chinese-Word-Vectors>项目下载解压后得到的TXT格式词向量文件
    - `--embedding_batch_size`：嵌入时的batch size，默认1024

`-ws`参数是分词方法，仅对需要分词的表征方法或模型有效：
- `jieba`：使用jieba包默认设置分词

`--max_sentence_length`：最长句长，默认512

# 3. 文本分类模型
通过`-m`参数传入模型名称，及可用的超参：
- `mlp`：线性分类器

各项超参（有些有的模型不能用）：
- `--optimizer`：默认`Adam`
- `--layer_num`：默认2
- `--hidden_dim`：默认128
- `--dropout`：默认0.5
- `--train_batch_size`：默认2048
- `--inference_batch_size`：默认4096
- `--cuda_device`：默认`cuda:0`

# 4. 代码运行模式
通过`-p`参数传入。
（本部分的`xx_metric`参数都和第五节的指标相同，但还可以添加loss（每个step损失函数的总和，不是准确的总loss值）
- `es`：（默认值）使用早停策略（只能应用于训练集+验证集+测试集都存在的场景）
    `--epoch_num`：最大epoch数
    `--train_metric`：每个epoch记录的训练集相应指标（会记录在wandb中）
    `--valid_metric`：每个epoch记录的验证集相应指标（会记录在wandb中）
    `--checkpoint_metric`：保留checkpoint所使用的指标（在valid_metric中的索引）
    `--patience`
    `--es_metric`：早停所使用的指标（在valid_metric中的索引），可连传多个（如果使用多个，则意为仅当多个指标都没有变好时，才应用早停机制）
- `ep`：固定模型运行总epoch数
    `--epoch_num`：固定epoch数


# 5. 实验结果和日志
指标名通过`--metric`传入，可连传多个：
- `acc`：准确率
- `macro-p`：macro-precision
- `macro-r`：macro-recall
- `macro-f1`：macro-F1