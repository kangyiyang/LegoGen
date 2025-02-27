import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pretrain_dataset_preprocess import Model_analysis
import time
import glob

working_path = os.getcwd()
import sys
sys.path.append(f'{working_path}/src')

from utils.utils import get_paths

# def is_rotation_matrix(matrix, tolerance=5e-3):
#     # 检查矩阵是否为 3x3 的二维数组
#     if not isinstance(matrix, np.ndarray) or matrix.shape != (3, 3):
#         return False

#     # 检查矩阵的转置是否等于逆矩阵
#     transpose = np.transpose(matrix)
#     try:
#         inverse = np.linalg.inv(matrix)
#     except np.linalg.LinAlgError:
#         return False
#     if not np.allclose(transpose, inverse, atol=tolerance):
#         return False

#     # 检查矩阵的行向量和列向量是否互相正交
#     for i in range(3):
#         for j in range(i + 1, 3):
#             if not np.isclose(np.dot(matrix[i], matrix[j]), 0, atol=tolerance):
#                 return False
#             if not np.isclose(np.dot(matrix[:, i], matrix[:, j]), 0, atol=tolerance):
#                 return False

#     # 检查矩阵的行向量和列向量的模是否为 1
#     for i in range(3):
#         if not np.isclose(np.linalg.norm(matrix[i]), 1, atol=tolerance):
#             return False
#         if not np.isclose(np.linalg.norm(matrix[:, i]), 1, atol=tolerance):
#             return False

#     return True





# not_rot_brick={}
# num_not_rot_model=0

# # min_s=100
# # max_s=0

# trans_array=np.zeros(6)

# good_rot=0
# num_rot=0
# most_rot=np.zeros(48)
# gt_rot=np.array(
#                 [
#                     #0---------------------
#                     [1,0,0, 0,1,0, 0,0,1],
#                     [1,0,0, 0,0,1, 0,-1,0],
#                     [1,0,0, 0,-1,0, 0,0,-1],
#                     [1,0,0, 0,0,-1, 0,1,0],

#                     [1,0,0, 0,1,0, 0,0,-1],
#                     [1,0,0, 0,0,1, 0,1,0],
#                     [1,0,0, 0,-1,0, 0,0,1],
#                     [1,0,0, 0,0,-1, 0,-1,0],

#                     #1---------------------
#                     [0,1,0, -1,0,0, 0,0,1],
#                     [0,1,0, 0,0,-1, -1,0,0],
#                     [0,1,0, 1,0,0, 0,0,-1],
#                     [0,1,0, 0,0,1, 1,0,0],

#                     [0,1,0, -1,0,0, 0,0,-1],
#                     [0,1,0, 0,0,-1, 1,0,0],
#                     [0,1,0, 1,0,0, 0,0,1],
#                     [0,1,0, 0,0,1, -1,0,0],

#                     #2---------------------
#                     [-1,0,0, 0,-1,0, 0,0,1],
#                     [-1,0,0, 0,0,1, 0,1,0],
#                     [-1,0,0, 0,1,0, 0,0,-1],
#                     [-1,0,0, 0,0,-1, 0,-1,0],

#                     [-1,0,0, 0,-1,0, 0,0,-1],
#                     [-1,0,0, 0,0,1, 0,-1,0],
#                     [-1,0,0, 0,1,0, 0,0,1],
#                     [-1,0,0, 0,0,-1, 0,1,0],

#                     #3---------------------
#                     [0,-1,0, 1,0,0, 0,0,1],
#                     [0,-1,0, 0,0,1, -1,0,0],
#                     [0,-1,0, -1,0,0, 0,0,-1],
#                     [0,-1,0, 0,0,-1, 1,0,0],

#                     [0,-1,0, 1,0,0, 0,0,-1],
#                     [0,-1,0, 0,0,1, 1,0,0],
#                     [0,-1,0, -1,0,0, 0,0,1],
#                     [0,-1,0, 0,0,-1, -1,0,0],
                    
#                     #4---------------------
#                     [0,0,1, 1,0,0, 0,1,0],
#                     [0,0,1, 0,1,0, -1,0,0],
#                     [0,0,1, -1,0,0, 0,-1,0],
#                     [0,0,1, 0,-1,0, 1,0,0],

#                     [0,0,1, 1,0,0, 0,-1,0],
#                     [0,0,1, 0,1,0, 1,0,0],
#                     [0,0,1, -1,0,0, 0,1,0],
#                     [0,0,1, 0,-1,0, -1,0,0],

#                     #5---------------------
#                     [0,0,-1, 1,0,0, 0,-1,0],
#                     [0,0,-1, 0,1,0, 1,0,0],
#                     [0,0,-1, -1,0,0, 0,1,0],
#                     [0,0,-1, 0,-1,0, -1,0,0],

#                     [0,0,-1, 1,0,0, 0,1,0],
#                     [0,0,-1, 0,1,0, -1,0,0],
#                     [0,0,-1, -1,0,0, 0,-1,0],
#                     [0,0,-1, 0,-1,0, 1,0,0],

#                     # # pad rot
#                     # [0,0,0, 0,0,0, 0,0,0],

#                 ]
#             )


model_paths=get_paths(Model_analysis.preprocessed_models_path)
new_bricks = os.listdir(Model_analysis.new_bricks_path)
original_bricks = os.listdir(Model_analysis.original_bricks_path)
bricks=new_bricks+original_bricks
bricks_obj=[brick.split('/')[-1][:-3]+'obj' for brick in bricks]

brick_meshs_path1 = glob.glob('dataset/bricks/complete_bricks_obj/meshes/*.obj')
brick_meshs1=[mesh.split('/')[-1] for mesh in brick_meshs_path1]
brick_meshs_path2 = glob.glob('dataset/bricks/export_obj/*.obj')
brick_meshs2=[mesh.split('/')[-1] for mesh in brick_meshs_path2]
brick_meshs=brick_meshs1+brick_meshs2

# bricks=[]
# for model_path in tqdm(model_paths,desc='Models and Bricks sorted'):
#     flag=0
#     with open(model_path,'r',encoding='utf-8') as file:
#         if_model=False
#         try:
#             model=file.readlines()
#         except:
#             print(f'无法读取文件{model_path}')
#         for i in range(len(model)):
#             line=model[i]
#             content=line.strip().split()
#             if not content:
#                 continue
#             if content[0]=='1':
#                 brick=' '.join(map(str, content[14:])).lower()
#                 brick=brick[:-3]+'obj'
#                 bricks.append(brick)

brick_meshs_set=set(brick_meshs)
bricks_set=set(bricks_obj)
diffs=list(bricks_set-brick_meshs_set)
diffs.sort(key=lambda x: x[0])

# models=[]
# for diff in diffs:
#     models.append(diff.split(' ')[0])

# models=list(set(models))
# models.sort(key=lambda x: x[0])
# print(len(models))
# print('#############################')
# for model in models:
#     print(model)

brick_list=[]
bricks_name={brick[:-4]:brick[-4:] for brick in bricks}
for diff in diffs:
    flag=False
    brick_name=diff[:-4]
    if brick_name in list(bricks_name.keys()):
        brick=brick_name+bricks_name[brick_name]
        brick_list.append(brick)

with open('dataset/bricks/brick_delete_list.txt','w',encoding='utf-8') as file:
    for brick in brick_list:
        file.write(brick)
        file.write('\n')
    #     brick_path=os.path.join(Model_analysis.new_bricks_path,brick)
    #     with open(brick_path,'r',encoding='utf-8') as file:
    #         model=file.readlines()
    #         for line in model:
    #             if line[0]=='5':
    #                 flag=True
    #                 break
    
    # if not flag:
    #     print(brick)


                # brick_1=brick.replace('\\', '_')
                # if brick not in brick_meshs:
                #     if '\\' in brick:
                #         brick=brick.replace('\\', '_')

                
                # rot=np.array([float(x) for x in content[5:14]])
                
                # for i in range(len(gt_rot)):
                #     if np.mean((rot - gt_rot[i]) ** 2)<1e-3:
                #         good_rot+=1
                #         most_rot[i]+=1
                #         break
                # num_rot+=1

                # trans=np.array([float(x) for x in content[2:5]])
                # if i>1:
                #     pre_line=model[i-1]
                #     pre_content=pre_line.strip().split()
                #     pre_trans=np.array([float(x) for x in pre_content[2:5]])
                #     trans1=trans[0]-pre_trans[0]
                #     trans2=trans[1]-pre_trans[1]
                #     trans3=trans[2]-pre_trans[2]
                # else:
                # trans1=trans[0]
                # trans2=trans[1]
                # trans3=trans[2]

                # if trans1<trans_array[0]:
                #     trans_array[0]=trans1
                #     flag=1
                # if trans1>trans_array[1]:
                #     trans_array[1]=trans1
                #     flag=1
                # if trans2<trans_array[2]:
                #     trans_array[2]=trans2
                #     flag=1
                # if trans2>trans_array[3]:
                #     trans_array[3]=trans2
                #     flag=1
                # if trans3<trans_array[4]:
                #     trans_array[4]=trans3
                #     flag=1
                # if trans3>trans_array[5]:
                #     trans_array[5]=trans3
                #     flag=1

                # Q, R = np.linalg.qr(rot)
                # scale=np.abs(np.diag(R))
                # if np.min(scale)<min_s:
                #     min_s=np.min(scale)
                #     print('min_s:',min_s)
                #     print('line:',line)
                # if np.max(scale)>max_s:
                #     max_s=np.max(scale)
                #     print('max_s:',max_s)
                #     print('line:',line)
                # if not is_rotation_matrix(rot):
                #     if_model=True
                #     brick=' '.join(map(str, content[14:])).lower()
                #     Model_analysis.brick_static_update(brick,not_rot_brick)
        # if if_model:
        #     num_not_rot_model+=1
    
    # if flag:
    #     print(trans_array)
    #     time.sleep(2)

# print(good_rot)
# print(num_rot)
# print(good_rot/num_rot)
# print(most_rot)

# print('min_x:',trans_array[0],'max_x:',trans_array[1])
# print('min_y:',trans_array[2],'max_y:',trans_array[3])
# print('min_z:',trans_array[4],'max_z:',trans_array[5])


# not_rot_brick_sorted=dict(sorted(not_rot_brick.items(),key=lambda x:x[1],reverse=True))
# not_rot_bricks=list(not_rot_brick_sorted.keys())
# num_not_rot_bricks=list(not_rot_brick_sorted.values())

# x_brick=np.arange(len(not_rot_bricks))+1
# print(f'The not_rot_models are {num_not_rot_model}')
# print(f'The not_rot_bricks are {len(not_rot_bricks)}')
# plt.loglog(x_brick,num_not_rot_bricks)
# plt.xlabel('brick_id')  # 横轴标签
# plt.ylabel('occurence')  # 纵轴标签
# plt.savefig('not_rot_bricks.png')
# plt.clf()

                    
