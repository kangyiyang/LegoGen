import torch
from torch.utils.data import Dataset
import os
import numpy as np
from itertools import product
from typing import List

# working_path = os.getcwd()
# import sys
# sys.path.append(f'{working_path}/src')

from utils.utils import content_to_position

class brick_tokenizer():
    def __init__(self) -> None:
        token_to_id_init=[
            # 同方向共42种=3*8+4*4+2*1
            [20.0, -24.0, -20.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [40.0, -24.0, -20.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [-60.0, -24.0, -20.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],

            [-20.0, -24.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, -24.0, 20.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [40.0, -24.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [60.0, -24.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],

            [0.0, -24.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],

            # 旋转方向共50种=4*8+4*4+2*1
            [-20.0, 24.0, 20.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [-20.0, -24.0, 40.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [40.0, -24.0, -20.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [40.0, -24.0, 40.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            
            [0.0, -24.0, -20.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, -24.0, -40.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [20.0, -24.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
            [40.0, -24.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],

            [0.0, -24.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0],

            # 逆旋转方向共50种=4*8+4*4+2*1
            [-20.0, 24.0, 20.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            [-20.0, -24.0, 40.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            [40.0, -24.0, -20.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            [40.0, -24.0, 40.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            
            [0.0, -24.0, -20.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            [0.0, -24.0, -40.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            [20.0, -24.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
            [40.0, -24.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],

            [0.0, -24.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],

        ]

        # TODO
        # tokenize的方式应该也会影响性能
        # 比如调整token id的顺序，或者将bos和eos按照最大的指定
        self.TOKEN_TO_ID={}
        id_count=3
        for one in token_to_id_init:
            trans_init=one[:3]
            combos = list(product([-1, 1], repeat=len(trans_init)))
            trans_all = [[a * b for a, b in zip(trans_init, combo)] for combo in combos]
            result_trans = [list(t) for t in set(tuple(sublist) for sublist in trans_all)]
            
            for trans in result_trans:
                key=tuple(trans+one[3:])
                self.TOKEN_TO_ID[key]=id_count
                id_count+=1
        
        self.ID_TO_TOKEN={value: key for key, value in self.TOKEN_TO_ID.items()}

        # 在这里bos_id就对应着'0 0 0 1 0 0 0 1 0 0 0 1'
        self.bos_id=2
        self.eos_id=1
        self.pad_id=0

        self.n_words=len(list(self.TOKEN_TO_ID.keys()))+3
        if self.n_words%8!=0:
            divisible_num=8-self.n_words % 8
            self.n_words+=divisible_num
        
        token_ids=list(self.TOKEN_TO_ID.keys())
        self.token_ids_array=np.array([list(x) for x in token_ids])

        self.begin_position=np.array([0,0,0,1,0,0,0,1,0,0,0,1])
    

    def tokenize(self, models:List[List[np.array]], chooses:List[List[int]], if_eos=True):
        models_tokenized=[]
        for model in models:
            model_tokenized=[self.bos_id]
            for content in model:
                mse=np.mean(np.square(np.subtract(content, self.token_ids_array)), axis=1)
                if np.min(mse)<1e-3:
                    model_tokenized.append(np.argmin(mse)+3)
            
            if if_eos:
                model_tokenized.append(self.eos_id)
            models_tokenized.append(model_tokenized)

        choose_eos=[self.eos_id] if if_eos else []
        chooses_tokenized=[[self.bos_id]+x+choose_eos for x in chooses]
            
        return models_tokenized, chooses_tokenized

    def detokenize(self, tokens:List[List[int]], chooses:List[List[int]], prompts: List[List[str]], gt_demo,in_tokens,in_chooses):
        bsz=len(tokens)
        generate_models_info=[]

        # if gt_demo:
        for i in range(bsz):
            model_info=['1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat'] if gt_demo else []
            for j in range(len(tokens[i])):
                try:
                    relative_position = content_to_position(self.ID_TO_TOKEN[tokens[i][j]])
                    # relative_position = content_to_position(self.ID_TO_TOKEN[in_tokens[i][j]])
                except:
                    # TODO 
                    # 关于choose的预测，采用mask可以指定对应的输出激活，其中bos和补全8带来的应该始终mask
                    # 而超出choose_max的也应该动态mask，训练和推理都需要这样
                    relative_position = content_to_position([0,0,0,1,0,0,0,1,0,0,0,1])
                
                choose=chooses[i][j]-3
                # choose=in_chooses[i][j]-3

                if gt_demo:
                    choose_max=j
                    choose = choose if choose<=choose_max else choose_max
                    choose_position = content_to_position(prompts[i][choose].strip().split()[2:14])
                else:
                    choose_max=j+len(prompts[i])
                    if choose<len(prompts[i]):
                        choose_position = content_to_position(prompts[i][choose].strip().split()[2:14])
                    elif choose<choose_max:
                        choose_position = content_to_position(model_info[choose-len(prompts[i])].strip().split()[2:14])
                    else:
                        if j!=0:
                            choose_position = content_to_position(model_info[j-1].strip().split()[2:14])
                        else:
                            choose_position = content_to_position(prompts[i][-1].strip().split()[2:14])
                
                position = choose_position @ relative_position

                trans_rot=list(position[:3,3])+list(position[:3,:3].reshape(-1))

                # print(in_tokens[i])
                # print(relative_position)
                # print(choose_position)
                # print(position)

                line_info=' '.join(map(str, [1,4]+trans_rot+['3001.dat']))
                model_info.append(line_info)

                # break

            generate_models_info.append(model_info)
            # break
        
        return generate_models_info