import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os
import json
from PIL import Image
from typing import List
import numpy as np

working_path = os.getcwd()
import sys
sys.path.append(f'{working_path}/src')

from utils.utils import get_paths, relative_transfer
from model.tokenizer import brick_tokenizer
from src.model.llama_model import ModelArgs
from src.model.clip import clip


class Simple_Dataset(Dataset):
    def __init__(self, opt, train=True):
        super().__init__()
        self.dataset_root=opt.dataset_root
        paths=get_paths(self.dataset_root)

        split_edge = opt.train_ratio*opt.data_size

        self.models=[]
        self.paths=[]
        for path in paths:
            with open(path,mode='r',encoding='utf-8') as file:
                data_id = int(path.split('/')[-1].split('_')[-1].split('.')[0])
                if data_id<=split_edge and train:
                    model=file.readlines()
                    self.models.append(model)
                    self.paths.append(path)
                elif data_id>split_edge and not train:
                    model=file.readlines()
                    self.models.append(model)
                    self.paths.append(path)

    def __getitem__(self, index):
        return self.models[index]
            
    def __len__(self):
        return len(self.models)


class Conditional_Dataset(Dataset):
    def __init__(self, opt, train=True):
        super().__init__()
        self.dataset_root=opt.dataset_root
        self.paths=get_paths(self.dataset_root)

        split_edge = opt.train_ratio*opt.data_size

        self.models=[]
        self.texts=[]
        for path in self.paths:
            with open(path,mode='r',encoding='utf-8') as file:
                model_name=path.split('/')[-1].split('_')
                data_id = int(model_name[-1].split('.')[0])
                text = model_name[0]

                model=file.readlines()
                if data_id<=split_edge and train:
                    self.models.append(model)
                    self.texts.append(text)
                elif data_id>split_edge and not train:
                    self.models.append(model)
                    self.texts.append(text)


    def __getitem__(self, index):
        return {
            'model':self.models[index],
            'text':self.texts[index]
            }
            
    def __len__(self):
        return len(self.models)



def collate_fn(data,opt,params: ModelArgs):
    bsz=len(data)
    tokenizer=brick_tokenizer()

    if opt.pretrain:
        models=data
    else:
        models=[x['model'] for x in data]
        texts=[x['text'] for x in data]

    relative_data=relative_transfer(models,tokenizer)
    relative_model=relative_data['model']
    relative_choose=relative_data['choose']

    models_id, chooses_id=tokenizer.tokenize(relative_model,relative_choose)
    
    min_prompt_len = min(len(t) for t in models_id)
    max_prompt_len = max(len(t) for t in models_id)
    if min_prompt_len<2:
        print('error:input can not be none')
    if max_prompt_len > params.max_seq_len:
        print('error:input seq is too long')
    
    # models_id的pad调整
    pad_id=tokenizer.pad_id
    tokens = torch.full((bsz, max_prompt_len), pad_id, dtype=torch.long,device='cpu')
    for k, t in enumerate(models_id):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long,device='cpu')

    chooses = torch.full((bsz, max_prompt_len), pad_id, dtype=torch.long,device='cpu')
    for k, t in enumerate(chooses_id):
        chooses[k, : len(t)] = torch.tensor(t, dtype=torch.long,device='cpu')

    collate_data = {
        'tokens':tokens,
        'chooses':chooses,
        'meta':{
            'min_prompt_len':min_prompt_len,
            'max_prompt_len':max_prompt_len,
            'pad_id':pad_id,
        }
    }

    if not opt.pretrain:
        collate_data['texts']=texts

    return collate_data