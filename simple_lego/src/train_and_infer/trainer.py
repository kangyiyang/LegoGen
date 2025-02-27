import os
import numpy as np
from typing import Dict
from fairscale.nn.model_parallel.initialize import get_model_parallel_rank

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

# from model.model import ModelArgs, Transformer
from ..model.llama_model import ModelArgs, Transformer
from ..utils.utils import seq_sample, sample_logits


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

        self.count=0

    def train_epoch(
                self,
                tokens: torch.Tensor,
                chooses: torch.Tensor,
                meta: Dict[str,int],
                texts: torch.Tensor = None,
                niu: float = 0.1,
                miu: float = 1,
                max_norm: float = 1,
                flat: bool = False,
                temperature: float = 0.6,
                top_p: float = 0.9,
                mix_ratio: float = 0,
        ):
        bsz, seq_len = tokens.shape

        self.model.train()
        with torch.autocast("cuda", enabled=True):
            if mix_ratio!=0:
                mix_token_logits, mix_choose_logits = self.model.forward(tokens, chooses, texts, start_pos=0)
                # tokens
                next_tokens = sample_logits(temperature,mix_token_logits,top_p)
                next_chooses = sample_logits(temperature,mix_choose_logits,top_p)
                
                # mix train
                random_prob = torch.rand(bsz, seq_len)<mix_ratio
                new_tokens = torch.where(random_prob, next_tokens, tokens)
                new_chooses = torch.where(random_prob, next_chooses, chooses)
                     
                token_logits, choose_logits = self.model.forward(new_tokens, new_chooses, texts, start_pos=0)
            else:
                token_logits, choose_logits= self.model.forward(tokens, chooses, texts, start_pos=0)
           
            token_logits_loss = token_logits[..., :-1, :].contiguous().view(-1, self.model.params.vocab_size)
            choose_logits_loss = choose_logits[..., :-1, :].contiguous().view(-1, self.model.params.max_seq_len)

            token_loss = F.cross_entropy(
                input=token_logits_loss,
                target=tokens[:, 1:].contiguous().view(-1),
                reduction="mean",
                ignore_index=meta['pad_id'],
            )

            choose_loss = F.cross_entropy(
                input=choose_logits_loss,
                target=chooses[:, 1:].contiguous().view(-1),
                reduction="mean",
                ignore_index=meta['pad_id'],
            )

            loss=miu*token_loss+niu*choose_loss
                    
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        self.optimizer.step()
        
        if not flat:
            self.scheduler.step()

        return {
            'loss':loss,
            'token_loss':miu*token_loss,
            'choose_loss':niu*choose_loss,
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
                temperature: float = 0.6,
                top_p: float = 0.9,
                accuracy: bool = True,
                scale: float = 1,
                threshold: float = 64,
                pretrain: bool = False,
        ):
        loss_all=[]
        token_loss_all=[]
        choose_loss_all=[]
        if accuracy:
            token_acc_all=[]
            choose_acc_all=[]
        
        self.model.eval()

        for i,data in enumerate(val_dataloader):
            tokens=data['tokens'].to(device)
            chooses=data['chooses'].to(device)
            if not pretrain:
                texts=data['texts']
            else:
                texts=None
            meta=data['meta']

            input_brick_mask = tokens[:,1:] != meta['pad_id']
            # pad_mask = tokens != meta['pad_id'] if bsz>1 else None
            
            token_logits, choose_logits = self.model.forward(tokens, chooses, texts, start_pos=0)
           
            token_logits_loss = token_logits[..., :-1, :].contiguous().view(-1, self.model.params.vocab_size)
            choose_logits_loss = choose_logits[..., :-1, :].contiguous().view(-1, self.model.params.max_seq_len)

            token_loss = F.cross_entropy(
                input=token_logits_loss,
                target=tokens[:, 1:].contiguous().view(-1),
                reduction="mean",
                ignore_index=meta['pad_id'],
            )

            choose_loss = F.cross_entropy(
                input=choose_logits_loss,
                target=chooses[:, 1:].contiguous().view(-1),
                reduction="mean",
                ignore_index=meta['pad_id'],
            )

            loss=miu*token_loss+niu*choose_loss

            loss_all.append(loss)
            token_loss_all.append(miu*token_loss)
            choose_loss_all.append(niu*choose_loss)

            if accuracy:
                # 这里计算acc将pad的都计算为正确的了，不太合适，所以指标会很高
                # 但现在batch_size都是1，所以没问题

                # brick_id的正确率
                next_tokens = sample_logits(temperature,token_logits,top_p,all=True)  
                # next_tokens:[batch_size,seq_len]
                total = next_tokens[:,:-1].numel()  # 总元素数量
                predict_tokens=next_tokens[:,:-1]*input_brick_mask
                correct = (predict_tokens == tokens[:,1:]).sum().item()  # 正确预测的数量
                # correct = (tokens[:,1:] == tokens[:,:-1]).sum().item()  # 正确预测的数量
                brick_acc = correct / total
                token_acc_all.append(brick_acc)

                # choose的正确率
                next_chooses = sample_logits(temperature,choose_logits,top_p,all=True)           
                # next_tokens:[batch_size,seq_len]
                total = next_chooses[:,:-1].numel()  # 总元素数量
                predict_chooses=next_chooses[:,:-1]*input_brick_mask
                correct = (predict_chooses == chooses[:,1:]).sum().item()  # 正确预测的数量
                # correct = (tokens[:,1:] == tokens[:,:-1]).sum().item()  # 正确预测的数量
                choose_acc = correct / total
                choose_acc_all.append(choose_acc)
                
        result={
            'loss':-torch.mean(torch.stack(loss_all)),
            'token_loss':torch.mean(torch.stack(token_loss_all)),
            'choose_loss':torch.mean(torch.stack(choose_loss_all)),
        }

        if accuracy:
            result['token_acc']=np.mean(token_acc_all)
            result['choose_acc']=np.mean(choose_acc_all)
            result['acc']=alpha*result['token_acc']+gamma*result['choose_acc']

        return result


    def freeze_model(self, model:Transformer,to_freeze_dict)->Transformer:
        for (name, param) in model.named_parameters():
            if name in to_freeze_dict:
                param.requires_grad = False
            elif name.split('.')[0]=='image_model' :
                param.requires_grad = False
            else:
                # print(name)
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
