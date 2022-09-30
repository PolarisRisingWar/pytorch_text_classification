def pad_list(v:list,max_length:int):
    """
    v是一个由未经pad的数值向量组成的列表
    返回值是pad后的向量和mask
    """
    if len(v)>=max_length:
        return (v[:max_length],[1 for _ in range(max_length)])
    else:
        padded_length=max_length-len(v)
        m=[1 for _ in range(len(v))]+[0 for _ in range(padded_length)]
        v.extend([0 for _ in range(padded_length)])
        return (v,m)