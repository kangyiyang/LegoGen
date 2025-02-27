cd ..

OMP_NUM_THREADS=1 \
python test.py \
    --test_input_dir 'simple_lego/demo/demo_result/3' \
    --baseline_model 'lego-DGMG/models/epoch_0200.h5' \
    --demo_acc \
    --baseline \
    --save_demo_acc \
    # --baseline_render \
    # --gt_demo \
    
    # --gt_demo \

