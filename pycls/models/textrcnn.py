#参考https://github.com/649453932/Chinese-Text-Classification-Pytorch/blob/master/models/TextRCNN.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextRCNN(nn.Module):
    def __init__(self,input_dim,output_dim,num_layers,dropout_rate,bias=True,bidirectional=True,hidden_dim=256):
        super(TextRCNN,self).__init__()

        num_directions=2 if bidirectional else 1

        self.rnns=nn.GRU(input_size=input_dim,hidden_size=hidden_dim,num_layers=num_layers,bias=bias,dropout=dropout_rate,bidirectional=bidirectional,
                        batch_first=True)
        self.maxpool = nn.MaxPool1d(input_dim)
        self.lin=nn.Linear(in_features=input_dim+num_directions*hidden_dim,out_features=output_dim)

    def forward(self,x:torch.Tensor):  #x是一个padding后的Tensor，seq_len是对应的长度
        op,hn=self.rnns(x)
        
        #hn: [num_layers * num_directions, batch_size, hidden_size]
        #op: [batch_size,max_sequence_length,hidden_dim*num_directions]

        op=torch.concat((x,op),2) #[batch_size, max_sequence_length, 特征加总]
        op = F.relu(op)
        op = op.permute(0, 2, 1)  #[batch_size, 特征加总, max_sequence_length]
        op = self.maxpool(op).squeeze()  #[batch_size, 特征加总, 1]
        op=self.lin(op)

        return op
