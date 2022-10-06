import torch.nn as nn

class MLP(nn.Module):
    def __init__(self,input_dim,output_dim,dropout_rate):
        super(MLP,self).__init__()

        self.dropout=nn.Dropout(dropout_rate)
        self.classifier=nn.Linear(input_dim,output_dim)
    
    def forward(self,x):
        x=self.dropout(x)
        x=self.classifier(x)

        return x