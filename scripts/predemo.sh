cd ..

OMP_NUM_THREADS=1 \
torchrun --nproc_per_node 2 --master_port=25647 demo.py \
    --max_seq_len 128 \
    --ckpt_dir 'val' \
    --max_batch_size 20 \
    --demo_input_dir 'demo/demo_input/3' \
    --demo_result_dir 'demo/demo_result/3' \
    --cuda \
    --USE_TENSORBOARD \
    --accuracy \
    --use_warm_up \
    --demo_acc \
    --pretrain \
    --save_demo_acc \
    # --gt_demo \