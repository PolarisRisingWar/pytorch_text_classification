#参考https://github.com/649453932/Chinese-Text-Classification-Pytorch/blob/master/models/TextRNN_Att.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU_Attention(nn.Module):
    def __init__(self,input_dim,output_dim,num_layers,dropout_rate,bias=True,bidirectional=True,hidden_dim=256):
        super(GRU_Attention,self).__init__()

        num_directions=2 if bidirectional else 1

        self.rnns=nn.GRU(input_size=input_dim,hidden_size=hidden_dim,num_layers=num_layers,bias=bias,dropout=dropout_rate,bidirectional=bidirectional,
                        batch_first=True)
        self.w=nn.Parameter(torch.zeros(hidden_dim*num_directions,1))
        self.lin=nn.Linear(in_features=hidden_dim*num_directions,out_features=output_dim)

        self.tanh1 = nn.Tanh()

    def forward(self,x:torch.Tensor,pad_list:torch.Tensor):  #x是一个padding后的Tensor，seq_len是对应的长度
        op,hn=self.rnns(x)
        #[batch_size,max_sequence_length,hidden_dim*num_directions]
        #用变长RNN的写法会导致pad出来的张量不等大

        M = self.tanh1(op)
        M=torch.matmul(M,self.w).squeeze(2)  #[batch_size, max_sequence_length, 1]
        
        #↓加mask版
        M=M*pad_list
        alpha=F.softmax(M,1).unsqueeze(2)  #[batch_size, max_sequence_length]

        out=op*alpha  #[batch_size, max_sequence_length, hidden_dim*num_directions]
        out=torch.sum(out,dim=1)  #[batch_size, hidden_dim*num_directions]

        out=self.lin(out)
        return out