import os

def load_iflytek(folder_path:str):
    """
    输入文件夹路径
    返回dict格式的数据集，键为train/val/test，train/val的值是dict（text是文本列表，label是标签列表），test的值是dict（id是标签列表，text是文本列表）
    """
    #训练集
    os.path.join(folder_path,'train.json')