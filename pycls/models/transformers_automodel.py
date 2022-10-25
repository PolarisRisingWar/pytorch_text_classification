import torch.nn as nn

from transformers import AutoModel

class TransformersAutoModel(nn.Module):
    def __init__(self,transformers_model_folder:str,input_dim,output_dim,dropout_rate):
        super(TransformersAutoModel,self).__init__()

        self.encoder=AutoModel.from_pretrained(transformers_model_folder)

        self.dropout=nn.Dropout(dropout_rate)
        self.classifier=nn.Linear(input_dim,output_dim)
    
    def forward(self,input_ids,token_type_ids,attention_mask):
        x=self.encoder(input_ids=input_ids,token_type_ids=token_type_ids,attention_mask=attention_mask)['pooler_output']
        x=self.dropout(x)
        x=self.classifier(x)

        return x