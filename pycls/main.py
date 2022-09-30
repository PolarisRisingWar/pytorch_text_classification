#命令行参数部分

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-dt","--dataset_type",help='预处理数据集的方法')
parser.add_argument("-df","--dataset_folder",help='数据集文件夹')

parser.add_argument("-e","--embedding_method",help='文本嵌入方法')

parser.add_argument("-ep","--embedding_model_path",help='文本嵌入预训练模型路径')
parser.add_argument("-et","--embedding_model_type",help='将文本嵌入预训练模型加载到本地的方法')
parser.add_argument("--embedding_batch_size",default=1024,type=int,help="嵌入时的batch size")

parser.add_argument("-ws","--word_segmentation",default="jieba",help='分词方法')

parser.add_argument("--max_sentence_length",default=512,type=int)

parser.add_argument("-m","--model",default='mlp',help="文本分类模型名称")

parser.add_argument("--optimizer",default="Adam")
parser.add_argument("--layer_num",default=2,type=int)
parser.add_argument("--hidden_dim",default=128,type=int)
parser.add_argument("--dropout",default=0.5,type=float)
parser.add_argument("--train_batch_size",default=2048,type=int)
parser.add_argument("--inference_batch_size",default=4096,type=int)
parser.add_argument("--cuda_device",default='cuda:0')

parser.add_argument("--epoch_num",default=10,type=int)

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
assert arg_dict['dropout']>=0 and arg_dict['dropout']<=1

if isinstance(arg_dict["metric"],str):
    arg_dict["metric"]=arg_dict['metric']

print(arg_dict)











###代码运行部分
import json,os,sys
from tqdm import tqdm
from datetime import datetime

import numpy as np

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from pycls.embedding_utils import load_w2v_matrix,pad_list

#文本表征部分
if arg_dict['embedding_method']=='w2v_mean':  #需要分词的表示方法，返回分词函数
    if arg_dict['word_segmentation']=='jieba':
        import jieba
        word_segmentation_function=jieba.lcut


if arg_dict['embedding_method']=='w2v_mean':  #word2vec系
    embedding_weight,word2id=load_w2v_matrix(arg_dict['embedding_model_path'],arg_dict['embedding_model_type'])
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


    #训练集
    train_data=[json.loads(x) for x in open(arg_dict['dataset_folder']+'/train.json').readlines()]
    train_text=[x['sentence'] for x in train_data]
    train_dataloader=DataLoader(train_text,arg_dict["embedding_batch_size"],shuffle=False,collate_fn=collate_fn)
    train_embedding=torch.zeros((len(train_text)),feature_dim)
    matrix_count=-1
    for batch in tqdm(train_dataloader):
        matrix_count+=1
        outputs=embedding(batch[0].to(arg_dict['cuda_device']))
        outputs=outputs.sum(axis=1)/batch[1].to(arg_dict['cuda_device']).sum(axis=1).unsqueeze(1)  #我显式把mask部分的嵌入置0了
        train_embedding[matrix_count*arg_dict["embedding_batch_size"]:matrix_count*arg_dict["embedding_batch_size"]+batch[0].size()[0]]=outputs

    #验证集
    dev_data=[json.loads(x) for x in open(arg_dict['dataset_folder']+'/dev.json').readlines()]
    dev_text=[x['sentence'] for x in dev_data]
    dev_dataloader=DataLoader(dev_text,arg_dict["embedding_batch_size"],shuffle=False,collate_fn=collate_fn)
    dev_embedding=torch.zeros((len(dev_text)),feature_dim)
    matrix_count=-1
    for batch in tqdm(dev_dataloader):
        matrix_count+=1
        outputs=embedding(batch[0].to(arg_dict['cuda_device']))
        outputs=outputs.sum(axis=1)/batch[1].to(arg_dict['cuda_device']).sum(axis=1).unsqueeze(1)  #我显式把mask部分的嵌入置0了
        dev_embedding[matrix_count*arg_dict["embedding_batch_size"]:matrix_count*arg_dict["embedding_batch_size"]+batch[0].size()[0]]=outputs

print(train_embedding.size())
print(dev_embedding.size())

#建立线性分类器
class LinearClassifier(nn.Module):
    def __init__(self,input_dim,output_dim=119):
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

#训练集
train_labels=torch.tensor([int(json.loads(x)['label']) for x in open(arg_dict['dataset_folder']+'/train.json').readlines()])
train_dataloader=DataLoader(TensorDataset(train_embedding,train_labels),batch_size=arg_dict['train_batch_size'],shuffle=True)
for epoch in tqdm(range(arg_dict['epoch_num']),desc='训练分类模型'):
    for batch in train_dataloader:
        model.train()
        optimizer.zero_grad()
        outputs=model(batch[0].to(arg_dict['cuda_device']))
        train_loss=loss_func(outputs,batch[1].to(arg_dict['cuda_device']))
        train_loss.backward()
        optimizer.step()

#验证集
dev_label=[int(json.loads(x)['label']) for x in open(arg_dict['dataset_folder']+'/dev.json').readlines()]
dev_predicts=[]
dev_dataloader=DataLoader(dev_embedding,batch_size=arg_dict['inference_batch_size'],shuffle=False)
with torch.no_grad():
    for batch in dev_dataloader:
        model.eval()
        outputs=model(batch.to(arg_dict['cuda_device']))
        dev_predicts.extend([i.item() for i in torch.argmax(outputs,1)])

#准确率 macro-precison macro-recall macro-F1
print(accuracy_score(dev_label,dev_predicts))
print(precision_score(dev_label,dev_predicts,average='macro'))
print(recall_score(dev_label,dev_predicts,average='macro'))
print(f1_score(dev_label,dev_predicts,average='macro'))