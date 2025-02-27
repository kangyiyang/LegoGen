cd ..

OMP_NUM_THREADS=1 \
torchrun --nproc_per_node 2 --master_port=25632 main.py \
    --pretrain_dataset_root 'dataset/pretrain_bricks dataset/processed_models' \
    --ckpt_dir 'llama_model/llama-2-13b' \
    --max_seq_len 512 \
    --lr 3e-4 \
    --niu 0.1 \
    --lamda 0.01 \
    --miu 0.1 \
    --max_mix 0.4 \
    --flat_lr_ratio 1 \
    --out_pad_dim 1 \
    --print_freq 100 \
    --val_freq 5 \
    --epochs 60 \
    --threshold 64 \
    --position_dim 128 \
    --cuda \
    --USE_TENSORBOARD \
    --accuracy \
    --use_warm_up \
    --demo_acc \
    --gt_demo \
    --pretrain \
    # --mix_train \