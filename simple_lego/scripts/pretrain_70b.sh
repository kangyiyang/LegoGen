cd ..

OMP_NUM_THREADS=1 \
/home/yyk/miniconda3/envs/lego/bin/torchrun --nproc_per_node 8 --master_port=25631 main.py \
    --dataset_root '/home/yyk/simple_lego/dataset/preprocessed_dataset' \
    --ckpt_dir '/NASdata/yyk/llama_model/llama-2-70b' \
    --max_seq_len 512 \
    --lr 4e-4 \
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