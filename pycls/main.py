#命令行参数部分

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-dt","--dataset_type",help='预处理数据集的方法',nargs='+')
parser.add_argument("-df","--dataset_folder",help='数据集文件夹')

parser.add_argument('-od','--output_dim',help='分类的标签数',type=int)

parser.add_argument("-e","--embedding_method",help='文本嵌入方法')
parser.add_argument("--pre_load",help="是否使用或储存本地的嵌入矩阵")
parser.add_argument("--embedding_folder",help="本地嵌入矩阵的存储路径")

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
parser.add_argument("--train_metric",nargs='*')
parser.add_argument("--valid_metric",nargs='*')
parser.add_argument("--checkpoint_metric",default=0,type=int)
parser.add_argument("--es_metric",default='0',nargs="+")

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
assert arg_dict['checkpoint_metric']>=0
assert arg_dict['dropout']>=0 and arg_dict['dropout']<=1

if isinstance(arg_dict["es_metric"],str):
    arg_dict["es_metric"]=[arg_dict['es_metric']]
arg_dict['es_metric']=[int(x) for x in arg_dict['es_metric']]
if isinstance(arg_dict["train_metric"],str):
    arg_dict["train_metric"]=[arg_dict['train_metric']]
if isinstance(arg_dict["valid_metric"],str):
    arg_dict["valid_metric"]=[arg_dict['valid_metric']]
if isinstance(arg_dict["metric"],str):
    arg_dict["metric"]=[arg_dict['metric']]
if isinstance(arg_dict['dataset_type'],str):
    arg_dict['dataset_type']=[arg_dict['dataset_type']]

print(arg_dict)











###代码运行部分
import json,os,sys
from tqdm import tqdm
from datetime import datetime
from copy import copy,deepcopy

import numpy as np

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from pycls.embedding_utils import load_w2v_matrix,pad_list
from pycls.dataset_utils import load_datasets

torch.autograd.set_detect_anomaly(True)

if arg_dict['wandb']:
    import wandb
    wandb.init(project="pytorch_text_cls",name=arg_dict['dataset_type'][0]+'_'+arg_dict['model']+'_'+str(datetime.now())[:10],config=arg_dict)

#导入数据
dataset_dict=load_datasets(arg_dict['dataset_type'],arg_dict['dataset_folder'])  #train/valid/test为键

if arg_dict['pre_load']=='load':
    for k in dataset_dict:
        dataset_dict[k]['embedding']=torch.load(os.path.join(arg_dict['embedding_folder'],k+'.pt'),map_location='cpu')
        feature_dim=dataset_dict[k]['embedding'].size()[-1]
        print(k+'嵌入的维度为：'+str(dataset_dict[k]['embedding'].size()))
        if arg_dict['embedding_method']=='w2v':
            dataset_dict[k]['pad_list']=torch.load(os.path.join(arg_dict['embedding_folder'],k+'_pad.pt'),map_location='cpu')
else:
    #文本表征部分
    #TODO: 速度太慢了，下次直接更新储存文本表征的功能吧
    if arg_dict['embedding_method'] in ['w2v','w2v_mean']:  #需要分词的表示方法，返回分词函数
        if arg_dict['word_segmentation']=='jieba':
            import jieba
            word_segmentation_function=jieba.lcut


    if arg_dict['embedding_method'] in ['w2v','w2v_mean']:  #word2vec系，需要分词
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

    if arg_dict['embedding_method'] in ['w2v','w2v_mean']:  #词表征系，embedding是将词嵌入为[sample_num,word_max_num,feature_dim]的PyTorch模型
        for k in dataset_dict:
            dataloader=DataLoader(dataset_dict[k]['text'],arg_dict["embedding_batch_size"],shuffle=False,collate_fn=collate_fn)
            if arg_dict['embedding_method']=='w2v': #不用池化
                temp_embedding=torch.zeros((len(dataset_dict[k]['text']),arg_dict['max_sentence_length'],feature_dim))
                temp_padlist=torch.zeros((len(dataset_dict[k]['text']),arg_dict['max_sentence_length']))
            else:  #需要池化
                temp_embedding=torch.zeros((len(dataset_dict[k]['text'])),feature_dim)
            matrix_count=-1
            for batch in tqdm(dataloader):
                matrix_count+=1
                outputs=embedding(batch[0].to(arg_dict['cuda_device']))
                if arg_dict['embedding_method']=='w2v':
                    temp_embedding[matrix_count*arg_dict["embedding_batch_size"]:matrix_count*arg_dict["embedding_batch_size"]+batch[0].size()[0],\
                                    :outputs.size()[1],:]=outputs
                    temp_padlist[matrix_count*arg_dict["embedding_batch_size"]:matrix_count*arg_dict["embedding_batch_size"]+batch[0].size()[0],\
                                    :outputs.size()[1]]=batch[1]
                elif arg_dict['embedding_method']=='w2v_mean':
                    outputs_sum=outputs.sum(axis=1)
                    outputs_num=torch.clamp(batch[1].to(arg_dict['cuda_device']).sum(axis=1),min=1)  #以防除0的情况出现 这玩意为0应该是空字符串吧
                    #这里有个问题在于，这个min不能是小于1的浮点数，因为保存不到int里面最后还是会变成0，我快被坑死了啊啊啊啊啊！
                    outputs=outputs_sum/outputs_num.unsqueeze(1)  #我显式把mask部分的嵌入置0了
                    temp_embedding[matrix_count*arg_dict["embedding_batch_size"]:matrix_count*arg_dict["embedding_batch_size"]+batch[0].size()[0]]=outputs
            dataset_dict[k]['embedding']=temp_embedding
            print('完成数据集'+k+'嵌入，其维度为：'+str(temp_embedding.size()))
            if arg_dict['embedding_method']=='w2v': #不用池化
                dataset_dict[k]['pad_list']=temp_padlist
                print('其pad list维度为：'+str(temp_padlist.size()))
            

if arg_dict['pre_load']=='save':
    for k in dataset_dict:
        torch.save(dataset_dict[k]['embedding'],os.path.join(arg_dict['embedding_folder'],k+'.pt'))
        print('已存储'+k+'嵌入到'+os.path.join(arg_dict['embedding_folder'],k+'.pt')+'位置！')
        if 'pad_list' in dataset_dict[k]:
            torch.save(dataset_dict[k]['pad_list'],os.path.join(arg_dict['embedding_folder'],k+'_pad.pt'))
            print('已存储'+k+'pad list到'+os.path.join(arg_dict['embedding_folder'],k+'_pad.pt')+'位置！')

#TODO: 感觉上面的内容应该把所有在GPU上的程序先下下来，再继续后面的代码

#建立分类器
if arg_dict['model']=='mlp':
    from pycls.models import MLP
    model=MLP(input_dim=feature_dim,output_dim=arg_dict['output_dim'],dropout_rate=arg_dict['dropout'])
if arg_dict['model']=='gru':
    from pycls.models import GRUEncoder
    model=GRUEncoder(input_dim=feature_dim,output_dim=arg_dict['output_dim'],num_layers=arg_dict['layer_num'],dropout_rate=arg_dict['dropout'])

model.to(arg_dict['cuda_device'])

optimizer=torch.optim.Adam(params=model.parameters(),lr=1e-4)
loss_func=nn.CrossEntropyLoss()

#早停和测试指标
metric_map={'acc':lambda y_true,y_pred:accuracy_score(y_true,y_pred),
            'macro-p':lambda y_true,y_pred:precision_score(y_true,y_pred,average='macro'),
            'macro-r':lambda y_true,y_pred:recall_score(y_true,y_pred,average='macro'),
            'macro-f1':lambda y_true,y_pred:f1_score(y_true,y_pred,average='macro')}

#训练集
if arg_dict['model']=='mlp':
    train_dataloader=DataLoader(TensorDataset(dataset_dict['train']['embedding'],torch.tensor(dataset_dict['train']['label'])),
                                batch_size=arg_dict['train_batch_size'],shuffle=True)
elif arg_dict['model']=='gru':
    train_dataloader=DataLoader(TensorDataset(dataset_dict['train']['embedding'],dataset_dict['train']['pad_list'],
                                torch.tensor(dataset_dict['train']['label'])),batch_size=arg_dict['train_batch_size'],shuffle=True)

#验证集
dev_label=dataset_dict['valid']['label']
if arg_dict['model']=='mlp':
    dev_dataloader=DataLoader(dataset_dict['valid']['embedding'],batch_size=arg_dict['inference_batch_size'],shuffle=False)
elif arg_dict['model']=='gru':
    dev_dataloader=DataLoader(TensorDataset(dataset_dict['valid']['embedding'],dataset_dict['valid']['pad_list']),
                            batch_size=arg_dict['inference_batch_size'],shuffle=False)

if arg_dict['running_mode']=='es':  #应用早停机制
    assert len(arg_dict['valid_metric'])>0  #就是说起码需要有验证指标才行
    assert arg_dict['checkpoint_metric']<len(arg_dict['valid_metric'])

    max_valid_metric=0  #用以衡量最终使用哪个epoch的checkpoint
    best_model={}
    accumulated_epoch=0  #早停积累的epoch数

    #训练集指标
    train_metrics=copy(arg_dict['train_metric'])
    if 'loss' in arg_dict['train_metric']:
        train_metrics.remove('loss')
        train_metric_loss=[]
    train_metrics_values=[]

    #验证集指标
    valid_metrics=copy(arg_dict['valid_metric'])
    if 'loss' in arg_dict['valid_metric']:
        valid_metrics.remove('loss')
        max_metrics_loss=float('inf')
    max_metrics=[0 for _ in valid_metrics]
    valid_metrics_values=[]

    for epoch in range(arg_dict['epoch_num']):
        #训练
        #TODO:其他train_metric指标
        if 'loss' in arg_dict['train_metric']:
            this_epoch_loss=0
        for batch in train_dataloader:
            model.train()
            optimizer.zero_grad()
            
            if arg_dict['model']=='mlp':  #输入是文本
                outputs=model(batch[0].to(arg_dict['cuda_device']))
                train_loss=loss_func(outputs,batch[1].to(arg_dict['cuda_device']))
            elif arg_dict['model']=='gru':
                outputs=model(batch[0].to(arg_dict['cuda_device']),batch[1].to(arg_dict['cuda_device']))
                train_loss=loss_func(outputs,batch[2].to(arg_dict['cuda_device']))

            
            if 'loss' in arg_dict['train_metric']:
                this_epoch_loss+=train_loss.item()
            train_loss.backward()
            optimizer.step()
            
        if 'loss' in arg_dict['train_metric']:
            train_metric_loss.append(this_epoch_loss)
        
        #验证
        dev_predicts=[]
        #TODO：验证集损失函数
        with torch.no_grad():
            for batch in dev_dataloader:
                model.eval()

                if arg_dict['model']=='mlp':  #输入是文本
                    outputs=model(batch.to(arg_dict['cuda_device']))
                elif arg_dict['model']=='gru':
                    outputs=model(batch[0].to(arg_dict['cuda_device']),batch[1].to(arg_dict['cuda_device']))

                dev_predicts.extend([i.item() for i in torch.argmax(outputs,1)])
        
        #记录指标
        this_epoch_metric=[metric_map[x](dev_label,dev_predicts) for x in valid_metrics]
        #print(this_epoch_metric)  #这个是拿来我每次测试代码时候用的
        valid_metrics_values.append(copy(this_epoch_metric))
        if arg_dict['wandb']:
            log_dict={valid_metrics[i]:this_epoch_metric[i] for i in range(len(valid_metrics))}
            log_dict['epoch']=epoch
            if 'loss' in arg_dict['train_metric']:
                log_dict['train_loss']=this_epoch_loss
            wandb.log(log_dict)

        if this_epoch_metric[arg_dict['checkpoint_metric']]>max_valid_metric:  #更新checkpoint
            max_valid_metric=this_epoch_metric[arg_dict['checkpoint_metric']]
            best_model=deepcopy(model.state_dict())

        #早停
        if all([this_epoch_metric[i]<max_metrics[i] for i in range(len(max_metrics))]):
            accumulated_epoch+=1
            if accumulated_epoch>=arg_dict['patience']:
                print('达到早停标准，停止程序运行')
                break
        else:
            accumulated_epoch=0
            max_metrics=[max(max_metrics[i],this_epoch_metric[i]) for i in range(len(max_metrics))]

        
#测试
model.load_state_dict(best_model)
test_label=dataset_dict['test']['label']
test_predicts=[]
if arg_dict['model']=='mlp':
    test_dataloader=DataLoader(dataset_dict['test']['embedding'],batch_size=arg_dict['inference_batch_size'],shuffle=False)
elif arg_dict['model']=='gru':
    test_dataloader=DataLoader(TensorDataset(dataset_dict['test']['embedding'],dataset_dict['test']['pad_list']),
                            batch_size=arg_dict['inference_batch_size'],shuffle=False)

with torch.no_grad():
    for batch in test_dataloader:
        model.eval()
        
        if arg_dict['model']=='mlp':  #输入是文本
            outputs=model(batch.to(arg_dict['cuda_device']))
        elif arg_dict['model']=='gru':
            outputs=model(batch[0].to(arg_dict['cuda_device']),batch[1].to(arg_dict['cuda_device']))

        test_predicts.extend([i.item() for i in torch.argmax(outputs,1)])

for metric in arg_dict['metric']:
    print(metric)
    print(metric_map[metric](test_label,test_predicts))

if arg_dict['wandb']:
    wandb.finish()