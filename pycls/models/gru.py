import torch
import torch.nn as nn

class GRUEncoder(nn.Module):
    def __init__(self,input_dim,output_dim,num_layers,dropout_rate,bias=True,bidirectional=True,hidden_dim=256):
        super(GRUEncoder,self).__init__()

        self.rnns=nn.GRU(input_size=input_dim,hidden_size=hidden_dim,num_layers=num_layers,bias=bias,dropout=dropout_rate,bidirectional=bidirectional,
                        batch_first=True)
        self.lin=nn.Linear(in_features=hidden_dim*2 if bidirectional else hidden_dim,out_features=output_dim)

    def forward(self,x,pad_list):  #x是一个padding后的Tensor，seq_len是对应的长度
        op,hn=self.rnns(x)
        op=torch.mul(op,pad_list.unsqueeze(2).repeat(1,1,op.size()[2]))  #不能直接使用op[pad_list==0]=0：会报in-place变化的错
        outputs_sum=op.sum(axis=1)
        outputs_num=torch.clamp(pad_list.sum(axis=1),min=1)
        outputs=outputs_sum/outputs_num.unsqueeze(1)
        outputs=self.lin(outputs)
        return outputs