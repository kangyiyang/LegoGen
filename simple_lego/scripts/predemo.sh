cd ..

OMP_NUM_THREADS=1 \
torchrun --nproc_per_node 2 --master_port=25641 demo.py \
    --dataset_root 'simple_lego/dataset/preprocessed_dataset' \
    --ckpt_dir 'val' \
    --demo_input_dir 'simple_lego/demo/demo_input/1' \
    --demo_result_dir 'simple_lego/demo/demo_result/1' \
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