import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from src.data_preprocess.pretrain_dataset_preprocess import Model_analysis

working_path = os.getcwd()
import sys
sys.path.append(f'{working_path}/src')

from utils.utils import get_paths

model_paths=get_paths(Model_analysis.preprocessed_models_path)

# 先得统计出来所有砖块的使用情况
# 定义一个字典，键是model，值是其用的砖块

new_bricks = os.listdir(Model_analysis.new_bricks_path)
original_bricks = os.listdir(Model_analysis.original_bricks_path)
new_bricks_num={}
original_bricks_num={}

for model_path in tqdm(model_paths,desc='Models and Bricks sorted'):
    with open(model_path,'r',encoding='utf-8') as file:
        try:
            model=file.readlines()
        except:
            print(f'无法读取文件{model_path}')
        for line in model:
            content=line.strip().split()
            if not content:
                continue
            if content[0]=='1':
                brick=' '.join(map(str, content[14:])).lower()

                if brick in original_bricks:
                    Model_analysis.brick_static_update(brick,original_bricks_num)
                elif brick in new_bricks:
                    Model_analysis.brick_static_update(brick,new_bricks_num)
                else:
                    if '\\' in brick:
                        brick=brick.replace('\\', '_')
                        
                        if brick in new_bricks:
                            Model_analysis.brick_static_update(brick,new_bricks_num)
                        else:
                            brick=brick.split('_')[1]
                            if brick in new_bricks:
                                Model_analysis.brick_static_update(brick,new_bricks_num)
                            else:
                                print(f'未知砖块:{brick}')
                    else:
                        print(f'未知砖块:{brick}')

original_bricks_num_sorted=dict(sorted(original_bricks_num.items(),key=lambda x:x[1],reverse=True))
less_original_bricks=original_bricks_num_sorted.keys()
num_original_sorted=list(original_bricks_num_sorted.values())
new_bricks_num_sorted=dict(sorted(new_bricks_num.items(),key=lambda x:x[1],reverse=True))
less_new_bricks=list(new_bricks_num_sorted.keys())
num_new_sorted=list(new_bricks_num_sorted.values())

# x_brick=np.arange(len(original_bricks))+1
# print(f'The original bricks used in omr dataset is {len(original_bricks)}')
# plt.loglog(x_brick,num_original_sorted)
# plt.xlabel('brick_id')  # 横轴标签
# plt.ylabel('occurence')  # 纵轴标签
# plt.savefig('original_bricks.png')
# plt.show()
# plt.clf()

# x_brick=np.arange(len(new_bricks))+1
# print(f'The new bricks used in omr dataset is {len(new_bricks)}')
# plt.loglog(x_brick,num_new_sorted)
# plt.xlabel('brick_id')  # 横轴标签
# plt.ylabel('occurence')  # 纵轴标签
# plt.show()
a=[x for x in original_bricks if x not in less_original_bricks]
print(a)
a=[x for x in less_original_bricks if x not in original_bricks]
print(a)

a=[x for x in new_bricks if x not in less_new_bricks]
print(a)
a=[x for x in less_new_bricks if x not in new_bricks]
print(a)