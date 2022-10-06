#命令行参数部分

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-dt","--dataset_type",help='预处理数据集的方法',nargs='+')
parser.add_argument("-df","--dataset_folder",help='数据集文件夹')

parser.add_argument('-od','--output_dim',help='分类的标签数',type=int)

parser.add_argument("-e","--embedding_method",help='文本嵌入方法')

parser.add_argument("-ep","--embedding_model_path",help='文本嵌入预训练模型路径')
parser.add_argument("-et","--embedding_model_type",help='将文本嵌入预训练模型加载到本地的方法')
parser.add_argument("--embedding_batch_size",default=1024,type=int,help="嵌入时的batch size")

parser.add_argument("-ws","--word_segmentation",default="jieba",help='分词方法')

parser.add_argument("--max_sentence_length",default=512,type=int,help='每一句最长可以输入到模型中的token数')

parser.add_argument("-m","--model",default='mlp',help="文本分类模型名称")

parser.add_argument("--optimizer",default="Adam")
parser.add_argument("--layer_num",default=2,type=int)
parser.add_argument("--hidden_dim",default=128,type=int)
parser.add_argument("--dropout",default=0.5,type=float)
parser.add_argument("--train_batch_size",default=2048,type=int)
parser.add_argument("--inference_batch_size",default=4096,type=int)
parser.add_argument("--cuda_device",default='cuda:0')

parser.add_argument('-p','--running_mode',default='es')
parser.add_argument("--epoch_num",default=10,type=int)
parser.add_argument("--patience",default=5,type=int)
parser.add_argument("--valid_metric",default="acc")
parser.add_argument("--es_metric",default="acc",nargs="+")

parser.add_argument("--metric",default="acc",nargs="+")

parser.add_argument('--wandb',action='store_true',help='是否开启wandb记录功能')

args = parser.parse_args()
arg_dict=args.__dict__

assert arg_dict['layer_num']>0
assert arg_dict['hidden_dim']>0
assert arg_dict["embedding_batch_size"]>0
assert arg_dict['train_batch_size']>0
assert arg_dict['inference_batch_size']>0
assert arg_dict['epoch_num']>0
assert arg_dict['patience']>0
assert arg_dict['dropout']>=0 and arg_dict['dropout']<=1

if isinstance(arg_dict["es_metric"],str):
    arg_dict["es_metric"]=[arg_dict['es_metric']]
if isinstance(arg_dict["metric"],str):
    arg_dict["metric"]=[arg_dict['metric']]
if isinstance(arg_dict['dataset_type'],str):
    arg_dict['dataset_type']=[arg_dict['dataset_type']]

print(arg_dict)











###代码运行部分
import json,os,sys
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy

import numpy as np

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from pycls.embedding_utils import load_w2v_matrix,pad_list
from pycls.dataset_utils import load_datasets

#导入数据
dataset_dict=load_datasets(arg_dict['dataset_type'],arg_dict['dataset_folder'])  #train/valid/test为键

#文本表征部分
#TODO: 速度太慢了，下次直接更新储存文本表征的功能吧
if arg_dict['embedding_method']=='w2v_mean':  #需要分词的表示方法，返回分词函数
    if arg_dict['word_segmentation']=='jieba':
        import jieba
        word_segmentation_function=jieba.lcut


if arg_dict['embedding_method']=='w2v_mean':  #word2vec系，需要分词
    embedding_weight,word2id=load_w2v_matrix(arg_dict['embedding_model_path'],arg_dict['embedding_model_type'])  #矩阵和词-索引对照字典
    feature_dim=embedding_weight.shape[1]

    embedding=nn.Embedding(embedding_weight.shape[0],feature_dim)
    embedding.weight.data.copy_(torch.from_numpy(embedding_weight))
    embedding.weight.requires_grad=False
    embedding.to(arg_dict['cuda_device'])

    def collate_fn(batch):
        jiebaed_text=[word_segmentation_function(sentence) for sentence in batch]  #每个元素是一个句子的列表，由句子中的词语组成

        mapped_text=[[word2id[word] if word in word2id else word2id['UNK'] for word in sentence] for sentence in jiebaed_text]
        #每个元素是一个句子的列表，由词语对应的索引组成

        max_len=min(arg_dict['max_sentence_length'],max([len(x) for x in mapped_text]))  #padding到的长度，限长
        padded_list=[pad_list(v,max_len) for v in mapped_text]

        numerical_text=torch.tensor([x[0] for x in padded_list])
        mask=torch.tensor([x[1] for x in padded_list])

        return (numerical_text,mask)

if arg_dict['embedding_method']=='w2v_mean':  #词表征系，embedding是将词嵌入为[sample_num,word_max_num,feature_dim]的PyTorch模型
    for k in dataset_dict:
        dataloader=DataLoader(dataset_dict[k]['text'],arg_dict["embedding_batch_size"],shuffle=False,collate_fn=collate_fn)
        temp_embedding=torch.zeros((len(dataset_dict[k]['text'])),feature_dim)
        matrix_count=-1
        for batch in tqdm(dataloader):
            matrix_count+=1
            outputs=embedding(batch[0].to(arg_dict['cuda_device']))
            outputs=outputs.sum(axis=1)/batch[1].to(arg_dict['cuda_device']).sum(axis=1).unsqueeze(1)  #我显式把mask部分的嵌入置0了
            temp_embedding[matrix_count*arg_dict["embedding_batch_size"]:matrix_count*arg_dict["embedding_batch_size"]+batch[0].size()[0]]=outputs
        dataset_dict[k]['embedding']=temp_embedding
        print('完成数据集'+k+'嵌入，其维度为：'+str(temp_embedding.size()))

#TODO: 感觉上面的内容应该把所有在GPU上的程序先下下来，再继续后面的代码

#建立线性分类器
class LinearClassifier(nn.Module):
    def __init__(self,input_dim,output_dim=arg_dict['output_dim']):
        super(LinearClassifier,self).__init__()

        self.dropout=nn.Dropout(arg_dict['dropout'])
        self.classifier=nn.Linear(input_dim,output_dim)
    
    def forward(self,x):
        x=self.dropout(x)
        x=self.classifier(x)

        return x

model=LinearClassifier(feature_dim)
model.to(arg_dict['cuda_device'])

optimizer=torch.optim.Adam(params=model.parameters(),lr=1e-4)
loss_func=nn.CrossEntropyLoss()

#早停和测试指标
metric_map={'acc':lambda y_true,y_pred:accuracy_score(y_true,y_pred),
            'macro-p':lambda y_true,y_pred:precision_score(y_true,y_pred,average='macro'),
            'macro-r':lambda y_true,y_pred:recall_score(y_true,y_pred,average='macro'),
            'macro-f1':lambda y_true,y_pred:f1_score(y_true,y_pred,average='macro')}

#训练集
train_dataloader=DataLoader(TensorDataset(dataset_dict['train']['embedding'],torch.tensor(dataset_dict['train']['label'])),
                            batch_size=arg_dict['train_batch_size'],shuffle=True)

#验证集
dev_label=dataset_dict['valid']['label']
dev_dataloader=DataLoader(dataset_dict['valid']['embedding'],batch_size=arg_dict['inference_batch_size'],shuffle=False)

if arg_dict['running_mode']=='es':  #应用早停机制
    best_model={}
    accumulated_epoch=0  #早停积累的epoch数
    max_valid_metric=0  #TODO：验证的这个可以算在早停的指标里面（如有重复）以减少计算量

    max_metrics=[0 for _ in arg_dict['es_metric']]  #TODO：loss
    es_metric_num=len(arg_dict['es_metric'])
    if 'loss' in arg_dict['es_metric']:  #负向的指标。目前只有loss，如果有别的以后再说
        loss_index=arg_dict['es_metric'].index('loss')
        max_metrics[loss_index]=float('inf')

    for epoch in tqdm(range(arg_dict['epoch_num']),desc='训练分类模型'):
        #训练
        for batch in train_dataloader:
            model.train()
            optimizer.zero_grad()
            outputs=model(batch[0].to(arg_dict['cuda_device']))
            train_loss=loss_func(outputs,batch[1].to(arg_dict['cuda_device']))
            train_loss.backward()
            optimizer.step()
        
        #验证
        dev_predicts=[]
        if 'loss' in arg_dict['es_metric']:
            dev_loss=0  #TODO：loss这部分我懒得写了，以后补
        with torch.no_grad():
            for batch in dev_dataloader:
                model.eval()
                outputs=model(batch.to(arg_dict['cuda_device']))
                dev_predicts.extend([i.item() for i in torch.argmax(outputs,1)])
        
        valid_metric=metric_map[arg_dict["valid_metric"]](dev_label,dev_predicts)
        if valid_metric>max_valid_metric:  #更新checkpoint
            max_valid_metric=valid_metric
            best_model=deepcopy(model.state_dict())

        
        #早停
        this_epoch_metric=[metric_map[x](dev_label,dev_predicts) for x in arg_dict['es_metric']]  #TODO：考虑loss
        #除loss以外的指标都赋值，loss位置赋0
        if 'loss' in arg_dict['es_metric']:  #TODO:负向的指标
            max_metrics[loss_index]=loss_func(outputs,batch[1].to(arg_dict['cuda_device']))

        if all([this_epoch_metric[i]<max_metrics[i] for i in range(es_metric_num)]):
            accumulated_epoch+=1
            if accumulated_epoch>=arg_dict['patience']:
                print('达到早停标准，停止程序运行')
        else:
            accumulated_epoch=0
            max_acc=accuracy_score(dev_label,dev_predicts)

        
#测试
model.load_state_dict(best_model)
test_label=dataset_dict['test']['label']
test_predicts=[]
test_dataloader=DataLoader(dataset_dict['test']['embedding'],batch_size=arg_dict['inference_batch_size'],shuffle=False)

with torch.no_grad():
    for batch in test_dataloader:
        model.eval()
        outputs=model(batch.to(arg_dict['cuda_device']))
        test_predicts.extend([i.item() for i in torch.argmax(outputs,1)])

for metric in arg_dict['metric']:
    print(metric)
    print(metric_map[metric](test_label,test_predicts))