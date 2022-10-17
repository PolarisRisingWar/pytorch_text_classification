import torch.nn as nn

from transformers import BertModel

from pycls.models import MLP

class Bert_Classifier(nn.Module):
    def __init__(self,transformers_model_folder:str,output_dim:int,dropout_rate:float=0.1) -> None:
        super(Bert_Classifier,self).__init__()

        encoder=BertModel.from_pretrained(transformers_model_folder)
        classifier=MLP(input_dim=768,output_dim=output_dim,dropout_rate=dropout_rate)
    
    def forward(self,x):
        
        return x





    