from functools import partial
import torch
from torch.utils.data import DataLoader

from transformers import get_cosine_schedule_with_warmup

from src.dataset import OMR_Pretrain_Dataset,Conditional_Dataset,collate_fn
from src.train_and_infer.generation import Llama
from src.train_and_infer.trainer import Trainer
from src.utils.options import opts


def main():
    print('setting options...')
    opt=opts().return_args()
    
    pretrain=opt.pretrain
    device = torch.device('cuda' if torch.cuda.is_available() and opt.cuda else 'cpu')

    print('loading model...')
    model_builed,checkpoint=Llama.build(
                    ckpt_dir=opt.ckpt_dir,
                    image_model_dir=opt.image_model_dir,
                    image_model_size=opt.image_model_size,
                    original_bricks_path=opt.original_bricks_path,
                    new_bricks_path=opt.new_bricks_path,
                    max_seq_len=opt.max_seq_len,
                    max_batch_size=opt.max_batch_size,
                    position_dim=opt.position_dim,
                    out_pad_dim=opt.out_pad_dim,
                    # out_rot_dim=opt.out_rot_dim,
                    rank=opt.rank,
                    c_n_heads=opt.c_n_heads,
                    patch_h=opt.patch_h,
                    patch_w=opt.patch_w,
                    image_dim=opt.image_dim,
                    add_cross=opt.add_cross,
                    seed=opt.seed,
                    pretrain=opt.pretrain,
                    )
    model=model_builed.model

    if pretrain:
        print('creating OMR_Pretrain_Dataset...')
        val_dataset=OMR_Pretrain_Dataset(opt,train=False)
    else:
        print('creating Conditional_Dataset...')
        val_dataset=Conditional_Dataset(opt, 'val')
    print(f'The size of dataset is {len(val_dataset)}')

    params = model.params
    assert opt.batch_size <= params.max_batch_size, (opt.batch_size, params.max_batch_size)

    val_dataloader = DataLoader(
                    val_dataset,
                    batch_size = 1,
                    shuffle = False,
                    pin_memory=True,
                    collate_fn=partial(collate_fn,opt=opt,params=params),
                    # sampler=train_sampler,
                    generator=torch.Generator(device = 'cuda')
                    )

    optimizer=torch.optim.AdamW(model.parameters())
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=opt.warm_up_ratio*100,
        num_training_steps=100,
    )

    print('building trainer')
    trainer=Trainer(to_freeze_dict=checkpoint,model=model,optimizer=optimizer,scheduler=lr_scheduler)
    
    print('start val...')
    log_dict_val=trainer.val_data(
                            device=device,
                            val_dataloader=val_dataloader, 
                            niu=opt.niu, 
                            lamda=opt.lamda, 
                            miu=opt.miu,
                            alpha=opt.alpha,
                            gamma=opt.gamma,
                            delta=opt.delta,
                            temperature=opt.temperature,
                            top_p=opt.top_p,
                            accuracy=opt.accuracy,
                            scale=opt.scale,
                            threshold=opt.threshold,
                            pretrain=opt.pretrain,
                            )

    message = ''
    for k, v in log_dict_val.items():
        message += '%s: %.7f ' % (k, abs(v))
    print(message)

if __name__ == "__main__":
    main()