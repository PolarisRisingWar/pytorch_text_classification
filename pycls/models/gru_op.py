#参考https://github.com/649453932/Chinese-Text-Classification-Pytorch/blob/master/models/TextRNN.py

import torch
import torch.nn as nn

class GRU_Output(nn.Module):
    def __init__(self,input_dim,output_dim,num_layers,dropout_rate,bias=True,bidirectional=True,hidden_dim=256):
        super(GRU_Output,self).__init__()

        self.rnns=nn.GRU(input_size=input_dim,hidden_size=hidden_dim,num_layers=num_layers,bias=bias,dropout=dropout_rate,bidirectional=bidirectional,
                        batch_first=True)
        self.lin=nn.Linear(in_features=hidden_dim*2 if bidirectional else hidden_dim,out_features=output_dim)

    def forward(self,x:torch.Tensor,pad_list:torch.Tensor):  #x是一个padding后的Tensor，seq_len是对应的长度
        lengths_list=[max(x,1) for x in pad_list.count_nonzero(1).tolist()]  #x等于0是空字符串情况，小于0是bug
        packed_input=nn.utils.rnn.pack_padded_sequence(x,lengths=lengths_list,batch_first=True,enforce_sorted=False)
        op,hn=self.rnns(packed_input)

        #hn:[num_layers * num_directions, batch_size, hidden_size]

        #seq_unpacked,lens_unpacked=nn.utils.rnn.pad_packed_sequence(op,batch_first=True)
        #如果batch_first=True，则：[batch_size,max_sequence_length,hidden_dim*num_directions]

        out=torch.cat((hn[-1],hn[-2]),-1).squeeze(0)  #[1,batch_size,num_directions*hidden_dim]
        out=self.lin(out)
        return out