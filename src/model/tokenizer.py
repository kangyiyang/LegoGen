import torch
from torch.utils.data import Dataset
import os
import numpy as np

working_path = os.getcwd()
import sys
sys.path.append(f'{working_path}/src')

from utils.utils import get_paths

class brick_tokenizer():
    def __init__(self,new_bricks_path,original_bricks_path) -> None:
        self.BRICK_TOKEN={}
        files = os.listdir(original_bricks_path)+os.listdir(new_bricks_path)
        for i in range(len(files)):
            self.BRICK_TOKEN[files[i]]=i+3
        
        self.TOKEN_BRICK={value: key for key, value in self.BRICK_TOKEN.items()}
        
        self.n_words=len(list(self.BRICK_TOKEN.keys()))+3
        if self.n_words%8!=0:
            divisible_num=8-self.n_words % 8
            self.n_words+=divisible_num
        
        self.bos_id=1
        self.eos_id=2
        self.pad_id=0
        
        self.bos_position=np.zeros(12)+1
        self.eos_position=np.zeros(12)-1
        self.pad_position=np.zeros(12)
    
    def convert_brick_to_id(self,content):
        brick=' '.join(map(str, content[14:])).lower()
        if brick in list(self.BRICK_TOKEN.keys()):
            return self.BRICK_TOKEN[brick]
        elif '\\' in brick:
            brick=brick.replace('\\', '_')
            if brick in list(self.BRICK_TOKEN.keys()):
                return self.BRICK_TOKEN[brick]
            else:
                brick=brick.split('_')[1]
                if brick in list(self.BRICK_TOKEN.keys()):
                    return self.BRICK_TOKEN[brick]
        else:
            print(f'未知砖块{brick}')
            return 0
    
    def devide_brick_and_postion(self,models,if_bos=True,if_eos=False,if_demo=False):
        # models:batch_size,seq_len
        models_id=[]
        models_position=[]
        for model_l in models:
            if not if_demo:
                model=model_l[0]
                if_bos=model_l[1]
                if_eos=model_l[2]
            else:
                model=model_l
            bricks_id,bricks_position=self.single_model_devide(model,if_bos,if_eos)
            models_id.append(bricks_id)
            models_position.append(bricks_position)
        
        return models_id,models_position
    
    def single_model_devide(self,model,if_bos=True,if_eos=False):
        bricks_id=[]
        bricks_position=[]
        if if_bos:
            bricks_id.append(self.bos_id)
            bricks_position.append(self.bos_position)   
        for brick in model:
            content=brick.strip().split()
            if content:
                bricks_id.append(self.convert_brick_to_id(content))
                bricks_position.append([float(x) for x in content[2:14]])
        if if_eos:
            bricks_id.append(self.eos_id)
            bricks_position.append(self.pad_position)
        
        return bricks_id,np.array(bricks_position)
    
    def get_colors(self, prompts, models_id, gt_demo):
        self.colors=[]
        if gt_demo:
            for prompt in prompts:
                colors=[]
                for line in prompt:
                    content=line.strip().split()
                    colors.append(int(content[1]))
                
                self.colors.append(colors)
        else:
            for i in range(len(models_id)):
                self.colors.append([16]*len(models_id[i]))

    def merge_brick_and_position(self,models_id,models_postion):
        models_info=[]
        for i,model_id in enumerate(models_id):
            model_info=self.single_model_merge(model_id,models_postion[i],self.colors[i])
            models_info.append(model_info)
        
        return models_info
        

    def single_model_merge(self,bricks_id, postions, color):
        bricks_info=[]
        for i, brick_id in enumerate(bricks_id):
            try:
                brick=self.TOKEN_BRICK[brick_id]
            except:
                brick='3020.dat'
            
            brick_info=' '.join(map(str, [1,color[i]]+postions[i].tolist()+[brick]))
            bricks_info.append(brick_info)
        
        return bricks_info