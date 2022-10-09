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

理论上倒是也可以自动计算，但是还没写，所以现在需要显式传入的参数：
`-od`：标签总数（即模型输出维度）
## 1.1 中文二分类
- ChnSentiCorp_htl_all.csv：CSV文件下载源<https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv>，处理方式为`ChnSentiCorp_htl_all`
## 1.2 中文multi-class单标签分类
- iflytek_public：压缩文件下载源<https://storage.googleapis.com/cluebenchmark/tasks/iflytek_public.zip>，处理方式为`iflytek`
## 1.3 中文multi-class multi-label分类
## 1.4 中文多任务multi-class单标签分类
## 1.5 其他文本分类数据集资料
1. [SophonPlus/ChineseNlpCorpus: 搜集、整理、发布 中文 自然语言处理 语料/数据集，与 有志之士 共同 促进 中文 自然语言处理 的 发展。](https://github.com/SophonPlus/ChineseNlpCorpus)：ChnSentiCorp_htl_all.csv出处
2. [文本分类（情感分析）中文数据集汇总 - 知乎](https://zhuanlan.zhihu.com/p/80029681)

# 2. 表征方法
通过`-e`参数传入方法名称
- 通过word2vec模型得到词表征后，直接离线得到样本表征：天然与后面模型部分解耦（本项目直接将文本嵌入部分前置了。预加载词表征可通过`--pre_load`参数实现，`save`值为储存当前的嵌入结果，`load`值为直接使用本地的嵌入结果，`--embedding_path`参数是储存路径。此外还有一边建模一边嵌入的写法，以后再写）。需要分词
    - `w2v`：得到word2vec表征（pad后的矩阵和seq_len）
    - `w2v_mean`：使用word2vec向量在样本上的平均值，作为样本表征
    - 预训练词向量的路径通过`-ep`参数传入
    - 加载预训练词向量的方法通过`-et`参数传入：
        - `Chinese-Word-Vectors`：<https://github.com/Embedding/Chinese-Word-Vectors>项目下载解压后得到的TXT格式词向量文件
    - `--embedding_batch_size`：嵌入时的batch size，默认1024

`-ws`参数是分词方法，仅对需要分词的表征方法或模型有效：
- `jieba`：使用jieba包默认设置分词

`--max_sentence_length`：最长句长，默认512

# 3. 文本分类模型
通过`-m`参数传入模型名称，及可用的超参：（有些模型具体的参数，如GRU是否双向等，懒得调了，以后再改吧）
- `mlp`：线性分类器
- `gru`：GRU（使用每一步隐藏层的平均池化作为最终的样本输出）
- `GRU_op`：GRU（使用最后一步隐藏层作为最终的样本输出）（本项目中用的是变长RNN）
- `GRU_att`：GRU+attention（注意这个attention理论上是带mask的，但是我没想好这个mask应该怎么实现。另一种实现方式可参考：[Apply mask softmax - PyTorch Forums](https://discuss.pytorch.org/t/apply-mask-softmax/14212/17)）
- `TextCNN`：TextCNN
- `TextRCNN`：TextRCNN

各项超参（有些有的模型不能用）：
- `--optimizer`：默认`Adam`
- `--lr`：默认1e-4
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
    `--es_metric`：早停所使用的指标（在valid_metric中的索引），可连传多个（如果使用多个，则意为仅当多个指标都没有好过之前的最好值时时，才应用早停机制）（←并不确定这种多个场景的操作是否合理）
- `ep`：固定模型运行总epoch数
    `--epoch_num`：固定epoch数


# 5. 实验结果和日志
指标名通过`--metric`传入，可连传多个：
- `acc`：准确率
- `macro-p`：macro-precision
- `macro-r`：macro-recall
- `macro-f1`：macro-F1

通过`--wandb`使用wandb来记录日志。本项目公开的wandb项目可见：<https://wandb.ai/afternoon/pytorch_text_cls>（由于项目持续改进，可能所使用的代码与最新的有所不同，但是基本可供参考）

项目中使用的各公开数据集上的结果可参考这篇石墨文档：<https://shimo.im/sheets/1lq7MaMyPBclrwAe/MODOC/>

# 6. 参考文献
1. torch所使用的随机种子默认值3407参考：[[2109.08203] Torch.manual_seed(3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision](https://arxiv.org/abs/2109.08203)（是不是很震惊于这玩意也能有参考文献？对，我也是这么想的）
2. TextCNN参考文献：[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
3. TextRNN/RNN系列参考文献：[Recurrent Neural Network for Text Classification with Multi-Task Learning](https://arxiv.org/abs/1605.05101)
4. TextRCNN参考文献：[Recurrent Convolutional Neural Networks for Text Classification](https://ojs.aaai.org/index.php/AAAI/article/view/9513/9372)

# 7. 其他参考资料
（有些代码针对性的参考资料放在了代码注释部分，本部分仅介绍比较通用的参考资料）
1. [Pytorch 是如何处理变长序列的 - 作业部落 Cmd Markdown 编辑阅读器](https://www.zybuluo.com/songying/note/1467532)
2. [pytorch中如何处理RNN输入变长序列padding - 知乎](https://zhuanlan.zhihu.com/p/34418001)
3. [中文文本分类 pytorch实现 - 知乎](https://zhuanlan.zhihu.com/p/73176084)