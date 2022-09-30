import os,json 

def load_datasets(dataset_name:str,folder_path:str):
    if dataset_name=='iflytek':
        return load_iflytek(folder_path)

def load_iflytek(folder_path:str):
    """
    输入文件夹路径
    返回dict格式的数据集，键为train/val/test，train/val的值是dict（text是文本列表，label是标签列表），test的值是dict（id是索引列表，text是文本列表）
    """
    dataset={}

    #训练集
    dataset['train']={}
    train_data=[json.loads(x) for x in open(os.path.join(folder_path,'train.json')).readlines()]
    dataset['train']['text']=[x['sentence'] for x in train_data]
    dataset['train']['label']=[int(x['label']) for x in train_data]

    #验证集
    dataset['valid']={}
    valid_data=[json.loads(x) for x in open(os.path.join(folder_path,'dev.json')).readlines()]
    dataset['valid']['text']=[x['sentence'] for x in valid_data]
    dataset['valid']['label']=[int(x['label']) for x in valid_data]

    #测试集
    dataset['test']={}
    test_data=[json.loads(x) for x in open(os.path.join(folder_path,'test.json')).readlines()]
    dataset['test']['text']=[x['sentence'] for x in test_data]
    dataset['test']['id']=[x['id'] for x in test_data]  #数字

    return dataset