#word2vec相关

import numpy as np

def load_w2v_matrix(embedding_path:str,embedding_type:str):
    """
    输入预训练模型路径和预训练模型类型
    输出numpy.ndarray格式的矩阵[词数,词向量维度] 和 word2id词典
    """
    if embedding_type=='Chinese-Word-Vectors':
        #1. 没有pad和UNK的表征，所以直接将索引0设置为pad表征（全0向量）,<UNK>表征则是所有嵌入的平均值
        embedding_list=[]
        embedding_list.append([0 for _ in range(300)])  #这个是pad的向量
        with open(embedding_path) as f:
            f_content=f.readlines()
        #第一行是嵌入的总词数和维度
        #从第二行开始，第一个空格之前的是词，后面的是向量（用空格隔开）

        f_content2=f_content[1:]
        first_space_index_list=[sentence.find(' ') for sentence in f_content2]
        word2id={f_content2[idx][:first_space_index_list[idx]]:idx+1 for idx in range(len(f_content2))}  #1到总词数
        nopad_embedding_list=[[float(x) for x in f_content2[idx][first_space_index_list[idx]:].split()] for idx in range(len(f_content2))]
        embedding_list.extend(nopad_embedding_list)

        #由于词向量中没有引入UNK，因此参考https://github.com/Embedding/Chinese-Word-Vectors/issues/74 用所有嵌入的平均值作为这一项值
        word2id['UNK']=len(f_content)  #0是pad的索引，所以已经有全的len(f_content)个词向量在了

        nopad_embedding_weight=np.array(nopad_embedding_list)
        unk_embedding=np.mean(nopad_embedding_weight,axis=0)

        embedding_list.append(unk_embedding.tolist())

        embedding_weight=np.array(embedding_list)

        return (embedding_weight,word2id)