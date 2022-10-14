#参考https://github.com/649453932/Chinese-Text-Classification-Pytorch/blob/master/models/Transformer.py（这个的实现更底层，且有其他细微差异）
#参考https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoder
#参考 pytorch nn.Transformer 的 mask 理解 https://www.cvmart.net/community/detail/4493
#参考 A detailed guide to PyTorch’s nn.Transformer() module. https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1

import numpy as np

import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self,input_dim,output_dim,dropout_rate=0.1) -> None:
        super(TransformerClassifier,self).__init__()

        self.input_dim=input_dim

        self.dropout=nn.Dropout(dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4,batch_first=True)  #input_dim/nhead必须是整数（变得奇怪起来了，以后再改吧）
        encoder_norm = nn.LayerNorm(input_dim, eps=1e-5)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,num_layers=6,norm=encoder_norm)
        #比较新的版本可以添加enable_nested_tensor=True
        #最后的layernorm我也不知道需不需要加其实

        self.fc=nn.Linear(input_dim,output_dim)
    
    def forward(self,x,pad_list):
        #x: [batch_size, max_sequence_length, embedding_dim]
        #pad_list: [batch_size, max_sequence_length]

        #positional encoding
        pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / self.input_dim)) for i in range(self.input_dim)] for pos in range(x.size()[1])])
        pe[:, 0::2] = np.sin(pe[:, 0::2])
        pe[:, 1::2] = np.cos(pe[:, 1::2])

        x+=pe.to(x.device)

        x=self.dropout(x)

        #key padding mask（不能有全0的情况出现，否则会返回NaN）
        bool_example=pad_list.bool()
        bool_example[bool_example.sum(1)==0]=1  #全0直接翻回全1：逻辑上应该也可以……
        kpm=~(bool_example)

        x=self.transformer_encoder(x,src_key_padding_mask=kpm)

        #其实感觉这个后面可以继续叠各种文本分类模型，但是，总之，此略，直接算个带mask的平均值：
        x=torch.mul(x,pad_list.unsqueeze(2).repeat(1,1,x.size()[2]))
        outputs_sum=x.sum(axis=1)
        outputs_num=torch.clamp(pad_list.sum(axis=1),min=1)
        outputs=outputs_sum/outputs_num.unsqueeze(1)
        outputs=self.fc(outputs)

        return outputs

