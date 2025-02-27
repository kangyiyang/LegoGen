from functools import partial
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import fire
import os

import torch
from torch import nn, Tensor
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.parallel import DistributedDataParallel as DDP

from fairscale.nn.model_parallel.initialize import get_model_parallel_rank
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from beartype import beartype
from beartype.door import is_bearable
from beartype.typing import Optional, Tuple, Type, List

from src.dataset import Simple_Dataset, Conditional_Dataset ,collate_fn
from src.train_and_infer.generation import Llama
from src.train_and_infer.trainer import Trainer
# from src.train_and_infer.conditional_trainer import C_Trainer
from src.utils.options import opts
from src.model.llama_model import Transformer
from src.utils.logger import Logger



def val(trainer: Trainer,device,opt,epoch,total_iters,logger:Logger,dataloader,path,metric):
    log_dict_val=trainer.val_data(
                                device=device,
                                val_dataloader=dataloader,
                                niu=opt.niu, 
                                lamda=opt.lamda, 
                                miu=opt.miu,
                                alpha=opt.alpha,
                                gamma=opt.gamma,  
                                temperature=opt.temperature,
                                top_p=opt.top_p,
                                accuracy=opt.accuracy,
                                scale=opt.scale,
                                threshold=opt.threshold,
                                pretrain=opt.pretrain,
                                )
    
    tr_va_data=os.path.basename(os.path.normpath(path))
    message = f'({tr_va_data}_epoch: {epoch}, iters: {total_iters})'
    logger.write('val_epoch: {} |'.format(epoch))

    for k, v in log_dict_val.items():
        logger.scalar_summary(f'{tr_va_data}_{k}', abs(v), epoch)
        logger.write('{} {:8f} | '.format(k, abs(v)))
        message += '%s: %.7f ' % (k, abs(v))
    print(message)

    if dict(log_dict_val.items())[opt.metric]>metric and tr_va_data=='val':
        print(f'saving the {tr_va_data}_best model (epoch {epoch}, total_iters {total_iters})')
        trainer.save_network(path,f'{tr_va_data}_best')
        metric=dict(log_dict_val.items())[opt.metric]
    
    return metric

def main():
    print('setting options...')
    opt=opts().return_args()

    pretrain=opt.pretrain
    device = torch.device('cuda' if torch.cuda.is_available() and opt.cuda else 'cpu')

    print('loading model...')
    model_builed,checkpoint=Llama.build(
                    ckpt_dir=opt.ckpt_dir,
                    text_model_size=opt.text_model_size,
                    # original_bricks_path=opt.original_bricks_path,
                    # new_bricks_path=opt.new_bricks_path,
                    max_seq_len=opt.max_seq_len,
                    max_batch_size=opt.max_batch_size,
                    position_dim=opt.position_dim,
                    out_pad_dim=opt.out_pad_dim,
                    # out_rot_dim=opt.out_rot_dim,
                    rank=opt.rank,
                    c_n_heads=opt.c_n_heads,
                    # patch_h=opt.patch_h,
                    # patch_w=opt.patch_w,
                    text_dim=opt.text_dim,
                    add_cross=opt.add_cross,
                    seed=opt.seed,
                    pretrain=opt.pretrain,
                    )
    model=model_builed.model
    # model_train=model_train.cuda(get_model_parallel_rank())
    # ddp_model=DDP(model_train,device_ids=[get_model_parallel_rank()])

    print('setting logger...')
    logger = Logger(opt)

    if opt.pretrain:
        print('creating Simple_Dataset...')
        train_dataset=Simple_Dataset(opt)
        val_dataset=Simple_Dataset(opt,False)
    else:
        print('creating Conditional_Dataset...')
        train_dataset=Conditional_Dataset(opt)
        val_dataset=Conditional_Dataset(opt,False)
    
    print(f'The size of dataset is {len(train_dataset)+len(val_dataset)}')
    # train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)

    params = model.params
    assert opt.batch_size <= params.max_batch_size, (opt.batch_size, params.max_batch_size)

    train_dataloader = DataLoader(
                    train_dataset,
                    batch_size = opt.batch_size,
                    shuffle = True,
                    drop_last = True,
                    pin_memory=True,
                    collate_fn=partial(collate_fn,opt=opt,params=params),
                    # sampler=train_sampler,
                    generator=torch.Generator(device = 'cuda')
                    )

    val_dataloader = DataLoader(
                    val_dataset,
                    batch_size = 1,
                    shuffle = False,
                    pin_memory=True,
                    collate_fn=partial(collate_fn,opt=opt,params=params),
                    # sampler=train_sampler,
                    generator=torch.Generator(device = 'cuda')
                    )

    print('initialize the optimizer...')
    optimizer=torch.optim.AdamW(
                model.parameters(),
                lr=opt.lr,
                betas=opt.betas,
                weight_decay=opt.weight_decay,
                eps=1e-5,
            )

    batch_num=len(train_dataloader)//opt.batch_size
    if len(train_dataloader)%opt.batch_size != 0:
        batch_num+=1
    whole_steps=batch_num * opt.epochs
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=opt.warm_up_ratio*whole_steps,
        num_training_steps=whole_steps,
    )

    
    print('building trainer')
    trainer=Trainer(to_freeze_dict=checkpoint,model=model,optimizer=optimizer,scheduler=lr_scheduler)

    print('starting training...')
    total_iters=0
    flat=False
    if opt.metric=='loss':
        train_metric=-float('inf')
        val_metric=-float('inf')
    elif opt.metric=='acc':
        train_metric=0
        val_metric=0

    if opt.mix_train:
        mix_ratio=np.arange(0,opt.max_mix,opt.max_mix/opt.epochs)
    else:
        mix_ratio=np.zeros(opt.epochs)

    for epoch in range(1,opt.epochs+1):
        logger.write('epoch: {} |'.format(epoch))
        for i,data in enumerate(train_dataloader):
            tokens=data['tokens'].to(device)
            chooses=data['chooses'].to(device)
            if pretrain:
                texts=None
            else:
                texts=data['texts']
            meta=data['meta']

            total_iters+=opt.batch_size

            if total_iters/whole_steps > opt.flat_lr_ratio:
                flat=True
            
            log_dict_train=trainer.train_epoch(
                                    tokens=tokens,
                                    meta=meta,
                                    chooses=chooses,
                                    texts=texts,
                                    niu=opt.niu, 
                                    miu=opt.miu,
                                    max_norm=opt.max_norm,
                                    flat=flat,
                                    mix_ratio=mix_ratio[epoch-1],
                                    )
            
            if total_iters % opt.print_freq == 0:
                message = '(epoch: %d, iters: %d) ' % (epoch, total_iters)
                for k, v in log_dict_train.items():
                    message += '%s: %.7f ' % (k, abs(v))
                print(message)
            
            if opt.save_by_iter and total_iters % opt.save_freq == 0:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = f'iter_{total_iters}'
                trainer.save_network(logger.save_model_path,save_suffix)
            
        for k, v in log_dict_train.items():
            logger.scalar_summary('train_{}'.format(k), abs(v), epoch)
            logger.write('{} {:8f} | '.format(k, abs(v)))
        
        if epoch % opt.val_freq==0:
            train_metric=val(trainer,device,opt,epoch,total_iters,logger,train_dataloader,logger.train_path,train_metric)
            val_metric=val(trainer,device,opt,epoch,total_iters,logger,val_dataloader,logger.val_path,val_metric)
    
        logger.write('\n')
    logger.close()    

    print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
    save_suffix = f'iter_{total_iters}(latest)'
    trainer.save_network(logger.latest_path,save_suffix)


if __name__ == "__main__":
    main()