cd ..

OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=5 /home/yyk/miniconda3/envs/lego/bin/torchrun --nproc_per_node 1 --master_port=25631 main.py \
    --dataset_root '/home/yyk/simple_lego/dataset/preprocessed_dataset' \
    --ckpt_dir '/NASdata/yyk/logs_2024-05-12-19-39/val' \
    --text_model_size 'ViT-B/32' \
    --rank 4 \
    --c_n_heads 40 \
    --text_dim 512 \
    --batch_size 1 \
    --add_cross  10\
    --lr 1.5e-4 \
    --miu 1 \
    --niu 1 \
    --max_mix 0.6 \
    --flat_lr_ratio 1 \
    --out_pad_dim 1 \
    --print_freq 40 \
    --val_freq 3 \
    --epochs 60 \
    --threshold 64 \
    --cuda \
    --USE_TENSORBOARD \
    --accuracy \
    --use_warm_up \
    --demo_acc \
    --gt_demo \
    # --mix_train \