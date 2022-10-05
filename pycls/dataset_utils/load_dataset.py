import os,json,csv,random

def load_datasets(dataset_names:list,folder_path:str):
    """
    输入文件夹路径及其他数据集预处理需要的参数
    返回dict格式的数据集，键为train/val/test
    train/val的值是dict（text是文本列表，label是标签列表）
    test的值是dict（id是索引列表（如有），text是文本列表，label是标签列表（如有））
    """
    if dataset_names==['iflytek']:
        return load_iflytek(folder_path)
    if dataset_names[0]=='ChnSentiCorp_htl_all':
        return load_ChnSentiCorp_htl_all(folder_path,dataset_names[1],dataset_names[2])

def load_iflytek(folder_path:str):
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

def load_ChnSentiCorp_htl_all(folder_path:str,random_seed:str,split_ratio:str):
    #加载数据
    with open(os.path.join(folder_path,'ChnSentiCorp_htl_all.csv')) as f:
        reader=csv.reader(f)
        header = next(reader)  #表头
        data = [[int(row[0]),row[1]] for row in reader]  #每个元素是一个由字符串组成的列表，第一个元素是标签（01），第二个元素是评论文本。
    
    random.seed(random_seed)
    random.shuffle(data)
    split_ratio_list=[int(i) for i in split_ratio.split('-')]  #TODO：这个其实也不一定要是int，但是这个以后再说吧
    split_point1=int(len(data)*split_ratio_list[0]/sum(split_ratio_list))
    split_point2=int(len(data)*(split_ratio_list[0]+split_ratio_list[1])/sum(split_ratio_list))
    train_data=data[:split_point1]
    valid_data=data[split_point1:split_point2]
    test_data=data[split_point2:]

    dataset={}
    dataset['train']={}
    dataset['train']['text']=[i[1] for i in train_data]
    dataset['train']['label']=[i[0] for i in train_data]

    dataset['valid']={}
    dataset['valid']['text']=[i[1] for i in valid_data]
    dataset['valid']['label']=[i[0] for i in valid_data]

    dataset['test']={}
    dataset['test']['text']=[i[1] for i in test_data]
    dataset['test']['label']=[i[0] for i in test_data]

    return dataset