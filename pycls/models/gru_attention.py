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
        self.w=nn.Parameter(torch.zeros(hidden_dim*num_directions))
        self.lin=nn.Linear(in_features=hidden_dim*num_directions,out_features=output_dim)

        self.tanh1 = nn.Tanh()

    def forward(self,x:torch.Tensor,pad_list:torch.Tensor):  #x是一个padding后的Tensor，seq_len是对应的长度
        lengths_list=[max(x,1) for x in pad_list.count_nonzero(1).tolist()]  #x等于0是空字符串情况，小于0是bug
        packed_input=nn.utils.rnn.pack_padded_sequence(x,lengths=lengths_list,batch_first=True,enforce_sorted=False)
        op,hn=self.rnns(packed_input)

        #hn:[num_layers * num_directions, batch_size, hidden_size]

        seq_unpacked,lens_unpacked=nn.utils.rnn.pad_packed_sequence(op,batch_first=True)
        #如果batch_first=True，则seq_unpacked：[batch_size,max_sequence_length,hidden_dim*num_directions]

        M = self.tanh1(seq_unpacked)
        M=torch.matmul(M,self.w).squeeze(2)  #[batch_size, max_sequence_length, 1]
        
        #↓加mask版
        M=M*pad_list
        alpha=F.softmax(M,1)  #[batch_size, max_sequence_length]

        out=seq_unpacked*alpha
        out=self.lin(out)
        return out