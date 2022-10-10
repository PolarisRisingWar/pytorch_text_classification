#参考https://github.com/649453932/Chinese-Text-Classification-Pytorch/blob/master/models/DPCNN.py
#还没看原论文和原始实现，感觉好像怪怪的。我跟这个参考代码的实现也不一样

import torch.nn as nn

class DPCNN(nn.Module):
    def __init__(self,input_dim,output_dim,num_filters=250) -> None:
        super(DPCNN,self).__init__()

        self.region=nn.Conv2d(1,num_filters,(3,input_dim))  #region embedding

        self.ewc=nn.Conv2d(num_filters,num_filters,3,padding='same')  #等长卷积（equal-width convolution）
        #padding='same'相当于使用nn.ZeroPad2d，但是我不确定pad方向是否也是对的，这个以后再看

        self.relu=nn.ReLU()

        self.maxpool=nn.MaxPool2d((3,1),stride=2,padding=(1,0))  #第3维稳定减半，第4维不变

        self.lin=nn.Linear(num_filters,output_dim)
    
    def forward(self,x):
        #region embedding
        x=self.region(x.unsqueeze(1))
        #[batch_size, max_sequence_length, input_dim]→[batch_size, 1, max_sequence_length, input_dim]→[batch_size, 256, max_sequence_length-2, 1]

        op=self.ewc(self.relu(x))
        op=self.ewc(self.relu(op))
        x+=op  #我参考的代码里没实现这个残差：https://github.com/649453932/Chinese-Text-Classification-Pytorch/issues/28

        while x.size()[2]>=2:
            x=self.maxpool(x)
            op=self.ewc(self.relu(x))
            op=self.ewc(self.relu(op))
            x+=op
            #print(x.size()[2])
        
        x=self.lin(x.squeeze())  #[batch_size, num_filters, 1]
        
        return x


