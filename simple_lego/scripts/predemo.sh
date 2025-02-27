cd ..

OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=6,7 /home/yyk/miniconda3/envs/lego/bin/torchrun --nproc_per_node 2 --master_port=25641 demo.py \
    --dataset_root '/home/yyk/simple_lego/dataset/preprocessed_dataset' \
    --ckpt_dir '/NASdata/yyk/logs_2024-04-28-20-56/val' \
    --demo_input_dir '/home/yyk/simple_lego/demo/demo_input/1' \
    --demo_result_dir '/home/yyk/simple_lego/demo/demo_result/1' \
    --max_seq_len 512 \
    --max_batch_size 40 \
    --cuda \
    --USE_TENSORBOARD \
    --accuracy \
    --use_warm_up \
    --demo_acc \
    --pretrain \
    --save_demo_acc \
    # --gt_demo \