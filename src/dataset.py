import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import os
import json
from PIL import Image

working_path = os.getcwd()
import sys
sys.path.append(f'{working_path}/src')

from utils.utils import get_paths, multi_seq, images_transform
from model.tokenizer import brick_tokenizer
from src.model.llama_model import ModelArgs


class OMR_Pretrain_Dataset(Dataset):
    def __init__(self, opt, train=True, demo=False):
        super().__init__()
        self.pretrain_dataset_root=opt.pretrain_dataset_root
        self.paths=get_paths(self.pretrain_dataset_root)

        train_paths, val_paths= train_test_split(self.paths, test_size=opt.val_ratio, random_state=opt.seed)
        val_paths, test_paths= train_test_split(val_paths, test_size=opt.val_ratio, random_state=opt.seed)

        train_models=multi_seq(train_paths,opt)
        val_models=multi_seq(val_paths,opt)

        if not demo:
            self.models=train_models if train else val_models
        else:
            self.test_paths=test_paths

    def __getitem__(self, index):
        return self.models[index]
            
    def __len__(self):
        return len(self.models)


class Conditional_Dataset(Dataset):
    def __init__(self, opt, split:str):
        super().__init__()
        conditional_path=opt.conditional_dataset_root
        models_path=os.path.join(conditional_path,f'models/{split}')
        images_path=os.path.join(conditional_path,f'images/{split}')
        
        self.model_files=get_paths(models_path)
        self.image_files=get_paths(images_path)

        self.models=[]
        self.images=[]

        models_split_json_path=os.path.join(opt.conditional_dataset_root,'model_split_num.json')
        with open(models_split_json_path, 'r') as f:
            model_split_info = json.load(f)

        for i in range(len(self.model_files)):
            with open(self.model_files[i],'r',encoding='utf-8') as file:
                model=file.readlines()

                model_name_whole=self.model_files[i].split('/')[-1]
                split_index=model_name_whole.rfind("_")
                model_name=model_name_whole[:split_index]
                model_id=int(model_name_whole[split_index+1])

                split_num=model_split_info[model_name]

                if split_num==1:
                    self.models.append([model,1,1])
                elif model_id==0:
                    self.models.append([model,1,0])
                elif model_id+1==split_num:
                    self.models.append([model,0,1])
                else:
                    self.models.append([model,0,0])

            image = Image.open(self.image_files[i]).convert('RGB')
            self.images.append(image)

    def __getitem__(self, index):
        return {
            'model':self.models[index],
            'image':self.images[index]
            }
            
    def __len__(self):
        return len(self.models)


def collate_fn(data,opt,params: ModelArgs):
    bsz=len(data)
    
    if opt.pretrain:
        bricks=data
    else:
        bricks=[x['model'] for x in data]
        images=[x['image'] for x in data]

    tokenizer=brick_tokenizer(opt.new_bricks_path,opt.original_bricks_path)
    models_id,models_position=tokenizer.devide_brick_and_postion(bricks)
    
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

    # models_position的pad调整
    pad_position=tokenizer.pad_position[0]
    tokens_position=torch.full((bsz, max_prompt_len, 12), pad_position, dtype=torch.float16,device='cpu')
    for k, t in enumerate(models_position):
        tokens_position[k, : len(t),:] = torch.tensor(t, dtype=torch.float16,device='cpu')

    collate_data = {
        'bricks_token':tokens,
        'bricks_position':tokens_position,
        'meta':{
            'min_prompt_len':min_prompt_len,
            'max_prompt_len':max_prompt_len,
            'pad_id':pad_id,
            'pad_position':pad_position
        }
    }

    if not opt.pretrain:
        images_transformed=[]
        for image in images:
            images_transformed.append(images_transform(image,opt))
        
        collate_data['images']=torch.stack(images_transformed)

    return collate_data