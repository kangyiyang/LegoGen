import os
import torch
import numpy as np
from typing import List
import torch.nn.functional as F
import torchvision.transforms as T

# from fairscale.nn.model_parallel.initialize import model_parallel_is_initialized

# from model.tokenizer import brick_tokenizer

#---------------------------------------------------------------
#---------------------------------------------------------------
# os utils
def get_paths(path)->List[str]:
    paths=[]
    files = os.listdir(path)
    for file in files:
        paths.append(os.path.join(path,file))

    return paths

def if_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

#---------------------------------------------------------------
#---------------------------------------------------------------
# pretrain_postprocess utils
def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def seq_sample(probs, p):
    """
    对每个 batch 进行 multinomial 采样的函数。
    """

    batch_size, seq_len, class_num = probs.size()

    # 创建结果张量
    sampled_tokens = torch.zeros(batch_size, seq_len, dtype=torch.long)

    # 对每个 batch 进行循环采样
    for i in range(batch_size):
        # 从概率分布中采样多个标记
        prob=probs[i]
        sampled_tokens[i] = sample_top_p(prob,p).reshape(-1)

    return sampled_tokens

def sample_logits(temperature,logits,top_p,all=False):
    log=logits[:,-1] if not all else logits

    if temperature > 0:
        probs = torch.softmax(log / temperature, dim=-1)
        if not all:
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = seq_sample(probs, top_p)
    else:
        next_token = torch.argmax(log, dim=-1)
    
    return next_token

#---------------------------------------------------------------
#---------------------------------------------------------------
# data utils
def relative_transfer(data:List[List[str]],tokenizer):
    new_models=[]
    new_choose=[]

    for model in data:
        new_model=[]
        choose=[]

        for i in range(1,len(model)):
            content=model[i].strip().split()[2:14]
            position=content_to_position(content)

            flag=False
            for j in np.arange(i)[::-1]:
                line=model[j]
                temp_content=line.strip().split()[2:14]
                temp_position=content_to_position(temp_content)

                transfer_position=np.linalg.inv(temp_position) @ position
                transfer_trans_rot=np.array(list(transfer_position[:3,3])+list(transfer_position[:3,:3].reshape(-1)))
                
                if np.min(np.mean(np.square(np.subtract(transfer_trans_rot, tokenizer.token_ids_array)), axis=1))<1e-3:
                    flag=True
                    choose.append(j+3)
                    new_model.append(transfer_trans_rot)
                    break
        
            if not flag:
                raise ValueError(f'no bricks connexted by previous: {model[:i+1]}')

        new_models.append(new_model)
        new_choose.append(choose)

    new_data={
        'model':new_models,
        'choose':new_choose,
    }

    return new_data


def content_to_position(content):
    trans_rot=[float(x) for x in content]
    position=np.eye(4)
    position[:3,:3]=np.array(trans_rot[3:]).reshape(3,3)
    position[:3,3]=trans_rot[:3]
    
    return position


#---------------------------------------------------------------
#---------------------------------------------------------------
# color utils
def hex_to_rgb(hex_code,alpha=1):
    # 去除可能存在的 '#' 字符
    hex_code = hex_code.strip('#')
    
    # 将16进制代码转换为RGB值
    r = int(hex_code[0:2], 16)
    g = int(hex_code[2:4], 16)
    b = int(hex_code[4:6], 16)
    
    rgb=np.array([r,g,b])
    if alpha!=1:
        rgb=np.int32(rgb+(1-alpha)*(256-rgb))

    # 返回RGB颜色值
    return rgb

def color_config(color_config_path):
    colors={}
    with open(color_config_path,'r',encoding='utf-8') as file:
        config=file.readlines()
        for line in config:
            if 'CODE' in line:
                content=line.strip().split()
                color_id=content[content.index('CODE')+1]
                color_hex=content[content.index('VALUE')+1]
                egde_color_hex=content[content.index('EDGE')+1]
                alpha=int(content[content.index('ALPHA')+1])/256 if 'ALPHA' in line else 1
                
                colors[color_id]={
                    'color':hex_to_rgb(color_hex,alpha),
                    'egde_color':hex_to_rgb(egde_color_hex,alpha)
                    }
    
    return colors