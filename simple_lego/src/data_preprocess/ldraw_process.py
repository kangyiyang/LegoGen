import os
import numpy as np

working_path = os.getcwd()
import sys
sys.path.append(f'{working_path}/src')

from utils.utils import if_exists, get_paths
from typing import List, Dict, Literal, Optional, Tuple

def ldraw_transfer(model:List[str],transfer_position):
    new_model=[]
    for line in model:
        content=line.strip().split()

        trans=np.array([int(x) for x in content[2:5]])
        rot=np.array([int(x) for x in content[5:14]])

        position=np.eye(4)
        position[:3,:3]=rot.reshape(3,3)
        position[:3,3]=trans
        new_position=transfer_position@position

        content[2:5]=new_position[:3,3]
        content[5:14]=new_position[:3,:3].reshape(-1)

        new_line=' '.join(map(str, content))

        new_model.append(new_line+'\n')

    return new_model

LdrawData_path=os.path.join(working_path,'dataset/ldraw_dataset')
preprocessed_dataset_path=os.path.join(working_path,'dataset/preprocessed_dataset')
if_exists(preprocessed_dataset_path)

ldraw_paths=get_paths(LdrawData_path)
# ldraw_names=list(set([path.split('/')[-1].split('_')[0] for path in ldraw_paths]))

# count=0
for path in ldraw_paths:
    with open(path,mode='r',encoding='utf-8') as ldraw_file:
        file_split=path.split('/')[-1].split('_')
        name=file_split[0]
        if name in ['random','2blocks-perpendicular'] or len(file_split)==3:
            continue
        
        ldraw_model=ldraw_file.readlines()
        
        content=ldraw_model[0].strip().split()

        trans=[int(x) for x in content[2:5]]
        rot=[int(x) for x in content[5:14]]

        if trans==[0,0,0] and rot==[1,0,0,0,1,0,0,0,1]:
            new_model=ldraw_model
            
        else:
            position=np.eye(4)
            position[:3,:3]=np.array(rot).reshape(3,3)
            position[:3,3]=trans

            transfer_position=np.linalg.inv(position)
            new_model=ldraw_transfer(ldraw_model,transfer_position)

        # count+=1
        new_path=os.path.join(preprocessed_dataset_path,f'{name}_{file_split[-1]}')
        with open(new_path,mode='w',encoding='utf-8') as new_file:
            new_file.writelines(new_model)
       


# print(count)
