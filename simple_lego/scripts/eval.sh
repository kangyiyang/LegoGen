cd ..

OMP_NUM_THREADS=1 \
torchrun --nproc_per_node 2 --master_port=25646 eval.py \
    --conditional_dataset_root 'lego/dataset/conditional_generation dataset' \
    --ckpt_dir 'val' \
    --image_model_dir 'lego/src/model/dinov2' \
    --image_model_size 'dinov2_vits14' \
    --rank 4 \
    --c_n_heads 40 \
    --image_dim 384 \
    --batch_size 1 \
    --patch_h 20 \
    --patch_w 20 \
    --add_cross 4 \
    --max_seq_len 512 \
    --niu 0.1 \
    --lamda 0.01 \
    --miu 0.1 \
    --flat_lr_ratio 1 \
    --out_pad_dim 1 \
    --threshold 64 \
    --cuda \
    --USE_TENSORBOARD \
    --accuracy \
    --use_warm_up \
    --demo_acc \
    --gt_demo \
    --pretrain \