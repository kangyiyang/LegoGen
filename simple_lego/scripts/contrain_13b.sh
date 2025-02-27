cd ..

OMP_NUM_THREADS=1 \
torchrun --nproc_per_node 2 --master_port=25631 main.py \
    --dataset_root 'simple_lego/dataset/preprocessed_dataset' \
    --ckpt_dir 'val' \
    --text_model_size 'ViT-B/32' \
    --rank 4 \
    --c_n_heads 40 \
    --text_dim 512 \
    --batch_size 1 \
    --add_cross  20\
    --lr 2e-4 \
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