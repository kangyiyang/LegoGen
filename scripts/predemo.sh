cd ..

OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=4,5 /home/yyk/miniconda3/envs/lego/bin/torchrun --nproc_per_node 2 --master_port=25647 demo.py \
    --max_seq_len 128 \
    --ckpt_dir '/NASdata/yyk/logs_2024-04-12-15-09/val' \
    --max_batch_size 20 \
    --demo_input_dir '/home/yyk/lego/demo/demo_input/3' \
    --demo_result_dir '/home/yyk/lego/demo/demo_result/3' \
    --cuda \
    --USE_TENSORBOARD \
    --accuracy \
    --use_warm_up \
    --demo_acc \
    --pretrain \
    --save_demo_acc \
    # --gt_demo \