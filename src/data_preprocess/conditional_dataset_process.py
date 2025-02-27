import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from glob import glob
import json

working_path = os.getcwd()
import sys
sys.path.append(f'{working_path}/src')

from utils.utils import if_exists,get_paths
from utils.render import model_Mesh_render
from utils.options import opts


class conditional_dataset():
    def __init__(self,opt) -> None:
        self.paths=get_paths(opt.pretrain_dataset_root)
        self.models_and_images_path=self.path_process(opt)
        self.model_split_json_path=os.path.join(opt.conditional_dataset_root,'model_split_num.json')
        self.render=model_Mesh_render(opt)
        

    def path_process(self,opt):
        split_paths=[]
        conditional_path=opt.conditional_dataset_root
        models_path=os.path.join(conditional_path,'models')
        images_path=os.path.join(conditional_path,'images')
        
        paths=[models_path,images_path]
        for path in paths:
            split_paths.append(os.path.join(path,'train'))
            split_paths.append(os.path.join(path,'val'))
            split_paths.append(os.path.join(path,'test'))
        
        for path in paths+split_paths:
            if_exists(path)

        return split_paths

    def models_process(self):
        train_paths, val_paths= train_test_split(self.paths, test_size=opt.val_ratio, random_state=opt.seed)
        val_paths, test_paths= train_test_split(val_paths, test_size=opt.val_ratio, random_state=opt.seed)
        
        source_paths=[train_paths,val_paths,test_paths]
        
        models_split={}
        for i in range(len(source_paths)):
            source_path=source_paths[i]
            new_path=self.models_and_images_path[i]
            for path in source_path:
                model_name=path.split('/')[-1][:-4]
                split_num=self.model_process(path, new_path, opt.max_seq_len-2)
                models_split[model_name]=split_num
        
        
        with open(self.model_split_json_path, 'w') as f:
            json.dump(models_split, f)

    def model_process(self,path,new_path,max_seq_len):
        new_models=[]
        file_name=path.split('/')[-1][:-4]
        file_suffix=path.split('/')[-1][-4:]
        with open(path,'r',encoding='utf-8') as file:            
            model=file.readlines()[1:]
            seq_len=len(model)

            if seq_len<=max_seq_len:
                new_models.append(model)
            else:
                multi_num=int(seq_len/max_seq_len)+1
                start=0
                end=max_seq_len

                while multi_num-1:
                    if model[start:end]==[]:
                        break
                    new_models.append(model[start:end])
                    start=end
                    end+=max_seq_len
                    multi_num-=1
                
                if model[start:end]!=[]:
                    new_models.append(model[start:end])

        for i in range(len(new_models)):
            with open(os.path.join(new_path,f'{file_name}_{i}{file_suffix}'),'w',encoding='utf-8') as file:
                for line in new_models[i]:
                    file.write(line)
        
        return len(new_models)

    def render_models(self):
        models_path=self.models_and_images_path[:3]
        images_path=self.models_and_images_path[3:]
        
        for i in range(len(models_path)):
            models_files=os.listdir(models_path[i])
            split_models=models_path[i].split('/')[-1]
            for model_file in tqdm(models_files,desc=f'{split_models} models rendering:'):
                name=model_file[:-4]
                model_path=os.path.join(models_path[i],model_file)
                with open(model_path,'r',encoding='utf-8') as file:
                    model=file.readlines()
                    save_image_file=os.path.join(images_path[i],f'{name}.png')
                    try:
                        render_image=self.render.visualize_model(model,save_image_file)
                    except:
                        print(f'render defeat:{model_path}')
                        # TODO
                        # 150-170\690-700\990-1000\1200-1300大约有超出face数量的



opt=opts().return_args()
Con_dataset=conditional_dataset(opt)
Con_dataset.models_process()
# Con_dataset.render_models()