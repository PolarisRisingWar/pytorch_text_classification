#代码实现和超参设置参考了https://github.com/649453932/Chinese-Text-Classification-Pytorch/blob/master/models/TextCNN.py

#理解模型可参考：
#https://zhuanlan.zhihu.com/p/40276005
#Convolutional Neural Networks for Sentence Classification | Let's Smile https://tw19941212.github.io/posts/f07060b1/

import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self,input_dim,output_dim,dropout_rate,filter_sizes=(2,3,4),num_filters=256) -> None:
        super(TextCNN,self).__init__()

        self.convs=nn.ModuleList([nn.Conv2d(1,num_filters,(k,input_dim)) for k in filter_sizes])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters * len(filter_sizes), output_dim)
    
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  #[batch_size,num_filters,H_{in}-1]
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  #[batch_size, num_filters]
        return x

    def forward(self,x:torch.Tensor):
        #x是经过pad后的词向量[batch_size, max_sentence_length, embedding_dim]
        x=x.unsqueeze(1)  #[batch_size, 1, max_sentence_length, embedding_dim]
        x=torch.cat([self.conv_and_pool(x,conv) for conv in self.convs], 1)  #[batch_size, len(filter_sizes)*num_filters]
        x = self.dropout(x)
        x = self.fc(x)
        return x