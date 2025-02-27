cd ..

OMP_NUM_THREADS=1 \
torchrun --nproc_per_node 2 --master_port=25645 demo.py \
    --ckpt_dir 'val' \
    --image_model_dir 'src/model/dinov2' \
    --image_model_size 'dinov2_vits14' \
    --demo_input_dir 'demo/demo_input/4' \
    --demo_result_dir 'demo/demo_result/4' \
    --rank 4 \
    --c_n_heads 40 \
    --image_dim 384 \
    --patch_h 20 \
    --patch_w 20 \
    --add_cross 4 \
    --max_batch_size 30 \
    --max_seq_len 128 \
    --cuda \
    --USE_TENSORBOARD \
    --accuracy \
    --use_warm_up \
    --demo_acc \
    # --gt_demo \