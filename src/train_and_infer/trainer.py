import json
import os
import sys
import time
from tqdm import tqdm
from pathlib import Path
import numpy as np
from typing import List, Dict, Literal, Optional, Tuple
from fairscale.nn.model_parallel.initialize import get_model_parallel_rank

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

# from model.model import ModelArgs, Transformer
from ..model.llama_model import ModelArgs, Transformer
from ..model.tokenizer import brick_tokenizer
from ..utils.utils import rotation_6d_to_9d, seq_sample, focal_loss, compute_box_vertices, calculate_iou, batch_qr_decomposition


class Trainer():
    def __init__(
                self,
                to_freeze_dict,
                model: Transformer,
                optimizer: AdamW,
                scheduler: LambdaLR,
        ):
        self.model=self.freeze_model(model,to_freeze_dict)
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.rot_choose=torch.tensor(
                    [
                        #0---------------------
                        [1,0,0, 0,1,0, 0,0,1],
                        [1,0,0, 0,0,1, 0,-1,0],
                        [1,0,0, 0,-1,0, 0,0,-1],
                        [1,0,0, 0,0,-1, 0,1,0],

                        #1---------------------
                        [0,1,0, -1,0,0, 0,0,1],
                        [0,1,0, 0,0,-1, -1,0,0],
                        [0,1,0, 1,0,0, 0,0,-1],
                        [0,1,0, 0,0,1, 1,0,0],

                        #2---------------------
                        [-1,0,0, 0,-1,0, 0,0,1],
                        [-1,0,0, 0,0,1, 0,1,0],
                        [-1,0,0, 0,1,0, 0,0,-1],
                        [-1,0,0, 0,0,-1, 0,-1,0],

                        #3---------------------
                        [0,-1,0, 1,0,0, 0,0,1],
                        [0,-1,0, 0,0,1, -1,0,0],
                        [0,-1,0, -1,0,0, 0,0,-1],
                        [0,-1,0, 0,0,-1, 1,0,0],

                        #4---------------------
                        [0,0,1, 1,0,0, 0,1,0],
                        [0,0,1, 0,1,0, -1,0,0],
                        [0,0,1, -1,0,0, 0,-1,0],
                        [0,0,1, 0,-1,0, 1,0,0],

                        #5---------------------
                        [0,0,-1, 1,0,0, 0,-1,0],
                        [0,0,-1, 0,1,0, 1,0,0],
                        [0,0,-1, -1,0,0, 0,1,0],
                        [0,0,-1, 0,-1,0, -1,0,0],

                        # # pad rot
                        # [0,0,0, 0,0,0, 0,0,0],

                    ], dtype=torch.float32
                )

        self.count=0

    def train_epoch(
                self,
                tokens: torch.Tensor,
                tokens_position: torch.Tensor,
                meta: Dict[str,int],
                images: torch.Tensor = None,
                niu: float = 0.1,
                lamda: float = 0.01,
                miu: float = 1,
                max_norm: float = 1,
                flat: bool = False,
                temperature: float = 0.6,
                top_p: float = 0.9,
                mix_ratio: float = 0
        ):
        bsz, seq_len = tokens.shape
        # input_position_mask = torch.all(torch.ne(tokens_position[:,1:], meta['pad_position']), dim=2).bool().view(-1,1)
        input_position_mask = ~torch.all(torch.eq(tokens_position[:,1:], meta['pad_position']), dim=2).bool().view(-1,1)
        
        self.model.train()
        with torch.autocast("cuda", enabled=True):
            if mix_ratio!=0:
                mix_logits, mix_position = self.model.forward(tokens, tokens_position, images, 0)
                # tokens
                if temperature > 0:
                    probs = torch.softmax(mix_logits / temperature, dim=-1)
                    next_tokens = seq_sample(probs, top_p)
                else:
                    next_tokens = torch.argmax(logits, dim=-1)
                # positions
                predicted_token_positions=mix_position.reshape(-1,9)

                trans=predicted_token_positions[:,:3]
                trans=trans+tokens_position[:,:,:3].reshape(-1,3)

                rot_6d=predicted_token_positions[:,3:]
                rot=rotation_6d_to_9d(rot_6d).view(-1,9)
                mse = torch.mean((rot.unsqueeze(1) - self.rot_choose.unsqueeze(0)) ** 2, dim=2)
                min_mse_indices = torch.argmin(mse, dim=1)
                rot_discrete=self.rot_choose[min_mse_indices]

                predicted_position=torch.cat((trans,rot_discrete),dim=1).reshape(bsz,-1,12)
                # mix train
                random_prob = torch.rand(bsz, seq_len)<mix_ratio
                new_tokens = torch.where(random_prob, next_tokens, tokens)
                new_tokens_position = torch.where(random_prob.unsqueeze(2), predicted_position, tokens_position)
                # new_tokens = torch.zeros(bsz,seq_len)
                # new_tokens_position = torch.zeros(bsz,seq_len,12)
                # for i in len(bsz):
                #     for j in len(seq_len):
                #         new_tokens[i,j]=next_tokens[i,j] if random_prob[i,j] else tokens[i,j]
                #         new_tokens_position[i,j]=predicted_position[i,j] if random_prob[i,j] else tokens_position[i,j]
                
                logits, position = self.model.forward(new_tokens, new_tokens_position, images, 0)
            else:
                logits, position = self.model.forward(tokens, tokens_position, images, 0)
            # logits_for_loss, trans, rot, gt_trans, gt_rot = self.result_process(logits,position,tokens_position)
            # token_loss, trans_loss, rot_loss, loss=self.caculate_loss(logits_for_loss, trans, rot, tokens, meta,input_position_mask,
            #                                                                     gt_trans, gt_rot, niu, lamda, miu)
            logits_for_loss, trans, rot, _, gt_relative_trans, _, gt_rot = self.result_process(logits,position,tokens_position)
            token_loss, trans_loss, rot_loss, loss=self.caculate_loss(logits_for_loss, trans, rot, tokens, meta,input_position_mask,
                                                                                gt_relative_trans, gt_rot, niu, lamda, miu)
            
            # logits, trans, rot = self.model.forward(tokens, tokens_position, 0)
            # logits_for_loss, trans_for_loss, rot_for_loss, _, gt_relative_trans, _, gt_rot_index, _ = self.result_process(
            #                                                                                                     logits, trans, rot, tokens_position)
            # token_loss, trans_loss, rot_loss, loss=self.caculate_loss(logits_for_loss, trans_for_loss, rot_for_loss, tokens, meta,input_position_mask,
            #                                                                     gt_relative_trans,gt_rot_index, niu, lamda, miu)

                    
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        self.optimizer.step()

        # self.count+=1
        # if self.count%50==0:
        #     self.check_parameters_updated()
        
        if not flat:
            self.scheduler.step()

        return {
            'loss':loss,
            'brick_loss':niu*token_loss,
            'trans_loss':lamda*trans_loss,
            'rot_loss':miu*rot_loss,
        }
    
    @torch.inference_mode
    def val_data(
                self,
                device,
                val_dataloader: DataLoader,
                niu: float = 0.1,
                lamda: float = 0.01,
                miu: float = 1,
                alpha: float = 0.4,
                gamma: float = 0.4,
                delta:float = 0.2,
                temperature: float = 0.6,
                top_p: float = 0.9,
                accuracy: bool = True,
                scale: float = 1,
                threshold: float = 64,
                pretrain: bool = False,
        ):
        loss_all=[]
        token_loss_all=[]
        trans_loss_all=[]
        rot_loss_all=[]
        if accuracy:
            brick_acc_all=[]
            position_iou_all=[]
            rot_similarity_all=[]
            trans_acc_all=[]
        
        self.model.eval()

        for i,data in enumerate(val_dataloader):
            tokens=data['bricks_token'].to(device)
            tokens_position=data['bricks_position'].to(device)
            if not pretrain:
                images=data['images'].to(device)
            else:
                images=None
            meta=data['meta']

            input_brick_mask = tokens[:,1:] != meta['pad_id']
            input_position_mask = ~torch.all(torch.eq(tokens_position[:,1:], meta['pad_position']), dim=2).bool().view(-1,1)

            logits, position = self.model.forward(tokens, tokens_position, images, 0)
           
            logits_for_loss, trans, rot, gt_trans, gt_relative_trans, predicted_trans, gt_rot, rot_discrete = self.result_process(logits,position,tokens_position,accuracy)
            token_loss, trans_loss, rot_loss, loss=self.caculate_loss(logits_for_loss, trans, rot, tokens, meta,input_position_mask,
                                                                                gt_relative_trans, gt_rot, niu, lamda, miu)

            loss_all.append(loss)
            token_loss_all.append(niu*token_loss)
            trans_loss_all.append(lamda*trans_loss)
            rot_loss_all.append(miu*rot_loss)

            if accuracy:
                # brick_id的正确率
                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_tokens = seq_sample(probs, top_p)
                else:
                    next_tokens = torch.argmax(logits, dim=-1)           
                # next_tokens:[batch_size,seq_len]
                total = next_tokens[:,:-1].numel()  # 总元素数量
                predict_tokens=next_tokens[:,:-1]*input_brick_mask
                correct = (predict_tokens == tokens[:,1:]).sum().item()  # 正确预测的数量
                # correct = (tokens[:,1:] == tokens[:,:-1]).sum().item()  # 正确预测的数量
                brick_acc = correct / total
                brick_acc_all.append(brick_acc)
                
                if not torch.all(input_position_mask == False):
                    # trans_mse
                    trans_mse = F.mse_loss(
                                    input=predicted_trans*input_position_mask,
                                    target=gt_trans[:,1:,:3].view(-1,3),
                                    reduction="none",
                                )
                    # trans_mse = F.mse_loss(
                    #                 input=gt_trans[:,:-1,:3].view(-1,3),
                    #                 target=gt_trans[:,1:,:3].view(-1,3),
                    #                 reduction="none",
                    #             )
                    # print('@@@@@@@@@@@@@@@@@@@')
                    # print(predicted_trans[:10])
                    # print(gt_trans[:,1:11,:3])
                    # print(trans_mse[:10])

                    count = torch.sum(trans_mse < threshold)
                    trans_acc = count / torch.numel(trans_mse)
                    trans_acc_all.append(trans_acc)

                    # rot similarity
                    rot_real=rot_discrete[input_position_mask.squeeze()]
                    rot_matrix=rot_real.view(rot_real.size(0),3,3)
                    gt_rot_real=gt_rot[input_position_mask.squeeze()]
                    gt_rot_matrix=gt_rot_real.view(gt_rot_real.size(0),3,3)

                    rot_similarity=torch.mean((F.cosine_similarity(rot_matrix,gt_rot_matrix,dim=2)+1)/2)
                    rot_similarity_all.append(rot_similarity)
        
        result={
            'loss':-torch.mean(torch.stack(loss_all)),
            'brick_loss':torch.mean(torch.stack(token_loss_all)),
            'trans_loss':torch.mean(torch.stack(trans_loss_all)),
            'rot_loss':torch.mean(torch.stack(rot_loss_all)),
        }

        if accuracy:
            result['brick_acc']=np.mean(brick_acc_all)
            result['trans_acc']=torch.mean(torch.stack(trans_acc_all))
            # result['position_iou']=torch.mean(torch.stack(position_iou_all))
            result['rot_similarity']=torch.mean(torch.stack(rot_similarity_all))
            result['acc']=alpha*result['brick_acc']+gamma*result['trans_acc']+delta*result['rot_similarity']
            # result['acc']=alpha*result['brick_acc']+gamma*result['position_iou']+delta*result['rot_similarity']

        return result

    def caculate_loss(self, logits_for_loss, trans, rot, tokens, meta,input_position_mask,
                      gt_trans, gt_rot, niu, lamda, miu):
        token_loss = F.cross_entropy(
                input=logits_for_loss,
                target=tokens[:, 1:].view(-1),
                reduction="mean",
                ignore_index=meta['pad_id'],
            )
        trans_loss = F.smooth_l1_loss(
                input=trans*input_position_mask,
                target=gt_trans*input_position_mask,
                reduction="mean",
            )
        # trans_loss = F.mse_loss(
        #         input=trans*input_position_mask,
        #         target=gt_trans*input_position_mask,
        #         reduction="mean",
        #     )
        # print('###################')
        # print(gt_trans[:10])
        # print(trans[:10])
        # test_loss = F.smooth_l1_loss(
        #         input=trans[:10],
        #         target=gt_trans[:10],
        #         reduction="none",
        #     )
        # print(test_loss)

        rot_loss = F.mse_loss(
                input=rot*input_position_mask,
                target=gt_rot,
                reduction="mean",
            )
       
        loss = niu*token_loss+lamda*trans_loss+miu*rot_loss
        # loss = lamda*trans_loss+miu*rot_loss
        if torch.isnan(loss):
            raise ValueError("nan loss encountered")
        
        return token_loss, trans_loss, rot_loss, loss


    def result_process(self, logits, position, tokens_position,accuracy=False):
        logits_for_loss = logits[..., :-1, :].contiguous().view(-1, self.model.params.vocab_size)
        # shape: (batch_size * seq_len, vocab_size)
        position_for_loss = position[..., :-1, :].contiguous().view(-1, 9)
        # shape: (batch_size * seq_len, 9)
        
        trans=position_for_loss[:,:3]
        predicted_trans=(position[:,:-1,:3]+tokens_position[:,:-1,:3]).view(-1,3)

        rot_6d=position_for_loss[:,3:]
        rot=rotation_6d_to_9d(rot_6d).view(-1,9)
        # rot=position_for_loss[:,3:]

        gt_rot=tokens_position[:,1:,3:].view(-1,9)
        # gt_trans=tokens_position[:,1:,:3].view(-1,3)
        gt_trans=tokens_position[:,:,:3]
        gt_relative_trans=(tokens_position[:,1:,:3]-tokens_position[:,:-1,:3]).view(-1,3)
        # print('-----------------')
        # print(gt_relative_trans[:10])
        # print(trans[:10])
        # print('------')
        # print(gt_trans[0,:10])
        # print(predicted_trans[:10])
        

        if accuracy:
            mse = torch.mean((rot.unsqueeze(1) - self.rot_choose.unsqueeze(0)) ** 2, dim=2)
            min_mse_indices = torch.argmin(mse, dim=1)
            rot_discrete=self.rot_choose[min_mse_indices]

        #     return logits_for_loss, trans, rot, gt_trans, gt_rot, rot_discrete
        # else:
        #     return logits_for_loss, trans, rot, gt_trans, gt_rot
            return logits_for_loss, trans, rot, gt_trans, gt_relative_trans, predicted_trans, gt_rot, rot_discrete
        else:
            return logits_for_loss, trans, rot, gt_trans, gt_relative_trans, predicted_trans, gt_rot


    def freeze_model(self, model:Transformer,to_freeze_dict)->Transformer:
        for (name, param) in model.named_parameters():
            if name in to_freeze_dict:
                param.requires_grad = False
            elif name.split('.')[0]=='image_model' :
                param.requires_grad = False
            else:
                pass
        
        return model
    
    def check_parameters_updated(self):
        # 检查参数是否发生变化
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is None:
                    print(f"Parameter '{name}' grad is None.")
                elif torch.any(param.grad != 0):
                    print(f"Parameter '{name}' has been updated.")
                else:
                    print(f"Parameter '{name}' grad is zero.")

    def save_network(self, save_dir:str ,save_suffix:str):
        model_parameters = self.model.state_dict()
        # 指定文件路径
        path = os.path.join(save_dir,f'{save_suffix}_{get_model_parallel_rank()}.pth')

        # 保存模型参数到文件
        torch.save(model_parameters, path)
