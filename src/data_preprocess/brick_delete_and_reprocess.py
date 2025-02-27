import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pretrain_dataset_preprocess import Model_analysis
import time
import glob
import shutil

working_path = os.getcwd()
import sys
sys.path.append(f'{working_path}/src')

from utils.utils import get_paths, if_exists

"""
注意由于删除后砖块的路径没有改变，因此这个文件只能执行一次，否则结果不正确
如果要重新运行，保证new_bricks,original_bricks都是没删除前的结果，这需要直接unzip new_bricks.zip/original_bricks.zip
而export_obj.zip需要在export_obj文件夹下解压
或者删除new和original bricks，然后重新运行pretrain_dataset_preprocess.py
"""

model_paths=get_paths(Model_analysis.preprocessed_models_path)
new_bricks = os.listdir(Model_analysis.new_bricks_path)
original_bricks = os.listdir(Model_analysis.original_bricks_path)
bricks=new_bricks+original_bricks
bricks_obj=[brick.split('/')[-1][:-3]+'obj' for brick in bricks]

brick_meshs_path1 = glob.glob('/home/yyk/lego/dataset/bricks/complete_bricks_obj/meshes/*.obj')
brick_meshs1=[mesh.split('/')[-1] for mesh in brick_meshs_path1]
brick_meshs_path2 = glob.glob('/home/yyk/lego/dataset/bricks/export_obj/*.obj')
brick_meshs2=[mesh.split('/')[-1] for mesh in brick_meshs_path2]
brick_meshs=brick_meshs1+brick_meshs2

brick_meshs_set=set(brick_meshs)
bricks_set=set(bricks_obj)
diffs=list(bricks_set-brick_meshs_set)

diffs.sort(key=lambda x: x[0])

brick_delete_list=[]
bricks_name={brick[:-4]:brick[-4:] for brick in bricks}
for diff in diffs:
    flag=False
    brick_name=diff[:-4]
    if brick_name in list(bricks_name.keys()):
        brick=brick_name+bricks_name[brick_name]
        brick_delete_list.append(brick)

#-----------------------------------------------------------------------
# bricks delete
for new_brick in new_bricks:
    if new_brick in brick_delete_list:
        brick_delete_path=os.path.join(Model_analysis.new_bricks_path,new_brick)
        try:
            os.remove(brick_delete_path)
        except:
            print(f'路径不存在：{brick_delete_path}')

new_bricks = os.listdir(Model_analysis.new_bricks_path)
original_bricks = os.listdir(Model_analysis.original_bricks_path)
print(f'original bricks now is {len(original_bricks)}')
print(f'new bricks now is {len(new_bricks)}')

# preprocessed models brick delete
preprocessed_models=os.listdir(Model_analysis.preprocessed_models_path)
processed_models_path='/home/yyk/lego/dataset/pretrain_bricks dataset/processed_models'

for model in preprocessed_models:
    processed_model=[]
    premodel_path=os.path.join(Model_analysis.preprocessed_models_path,model)
    model_path=os.path.join(processed_models_path,model)
    with open(premodel_path,'r',encoding='utf-8') as file:
        lines=file.readlines()
        for line in lines[1:]:
            content=line.strip().split()
            brick=' '.join(map(str, content[14:])).lower()
            if brick not in brick_delete_list:
                processed_model.append(line)
    
    with open(model_path,'w',encoding='utf-8') as model_file:
        model_file.write(lines[0])
        for line in processed_model:
            model_file.write(line)

#-----------------------------------------------------------------------
# objs合并
brick_objs_path='/home/yyk/lego/dataset/bricks/brick_objs'
if_exists(brick_objs_path)


for mesh in list(brick_meshs_set):
    if mesh in brick_meshs2:
        shutil.copy2(f'/home/yyk/lego/dataset/bricks/export_obj/{mesh}', os.path.join(brick_objs_path,mesh))
    elif mesh in brick_meshs1:
        shutil.copy2(f'/home/yyk/lego/dataset/bricks/complete_bricks_obj/meshes/{mesh}', os.path.join(brick_objs_path,mesh))
    else:
        print(f'error:unknown obj {mesh}')

brick_objs = os.listdir(brick_objs_path)
print(f'brick_objs now is {len(brick_objs)}')
