cd ..

OMP_NUM_THREADS=1 \
torchrun --nproc_per_node 2 --master_port=25633 main.py \
    --conditional_dataset_root 'dataset/conditional_generation dataset' \
    --ckpt_dir 'val' \
    --image_model_dir 'src/model/dinov2' \
    --image_model_size 'dinov2_vits14' \
    --rank 4 \
    --c_n_heads 40 \
    --image_dim 384 \
    --batch_size 1 \
    --patch_h 20 \
    --patch_w 20 \
    --add_cross 4 \
    --lr 2e-5 \
    --niu 0.1 \
    --lamda 0.01 \
    --miu 0.1 \
    --max_mix 0.6 \
    --flat_lr_ratio 1 \
    --out_pad_dim 1 \
    --print_freq 100 \
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