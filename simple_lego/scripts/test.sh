cd ..

OMP_NUM_THREADS=1 \
python test.py \
    --test_input_dir '/home/yyk/simple_lego/demo/demo_result/3' \
    --baseline_model '/NASdata/yyk/lego-DGMG/15-05-2024--13-37-16/models/epoch_0200.h5' \
    --demo_acc \
    --baseline \
    --save_demo_acc \
    # --baseline_render \
    # --gt_demo \
    
    # --gt_demo \

