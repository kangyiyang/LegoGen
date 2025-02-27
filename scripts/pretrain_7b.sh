cd ..

OMP_NUM_THREADS=1 \
torchrun --nproc_per_node 1 --master_port=25641 main.py \
    --max_seq_len 512 \
    --ckpt_dir 'llama_model/llama-2-7b' \
    --lr 1e-5 \
    --niu 1 \
    --lamda 0.1 \
    --miu 0.1 \
    --flat_lr_ratio 1 \
    --out_pad_dim 1 \
    --print_freq 100 \
    --val_freq 5 \
    --epochs 60 \
    --threshold 64 \
    --pretrain \
    --cuda \
    --USE_TENSORBOARD \
    --accuracy \
    --use_warm_up \
    --demo_acc \
    --gt_demo \
    --pretrain \