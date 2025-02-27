cd ..

OMP_NUM_THREADS=1 \
/home/yyk/miniconda3/envs/lego/bin/torchrun --nproc_per_node 8 --master_port=25640 demo.py \
    --dataset_root '/home/yyk/simple_lego/dataset/preprocessed_dataset' \
    --ckpt_dir '/NASdata/yyk/logs_2024-05-22-14-48/val' \
    --demo_input_dir '/home/yyk/simple_lego/demo/demo_input/3' \
    --demo_result_dir '/home/yyk/simple_lego/demo/demo_result/3' \
    --text_model_size 'ViT-B/32' \
    --max_seq_len 512 \
    --rank 4 \
    --c_n_heads 40 \
    --text_dim 512 \
    --patch_h 20 \
    --patch_w 20 \
    --add_cross 20 \
    --max_batch_size 40 \
    --cuda \
    --USE_TENSORBOARD \
    --accuracy \
    --use_warm_up \
    --demo_acc \
    # --gt_demo \