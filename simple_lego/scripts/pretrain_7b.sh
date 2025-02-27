cd ..

OMP_NUM_THREADS=1 \
torchrun --nproc_per_node 1 --master_port=25641 main.py \
    --dataset_root 'simple_lego/dataset/preprocessed_dataset' \
    --ckpt_dir 'llama_model/llama-2-7b' \
    --max_seq_len 512 \
    --lr 3e-4 \
    --miu 1 \
    --niu 1 \
    --max_mix 0.4 \
    --flat_lr_ratio 1 \
    --out_pad_dim 1 \
    --print_freq 40 \
    --val_freq 3 \
    --epochs 60 \
    --batch_size 1 \
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