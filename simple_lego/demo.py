from typing import List
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import shutil
from PIL import Image

from src.train_and_infer.generation import Llama
from src.utils.options import opts
from src.utils.utils import if_exists
from src.dataset import Simple_Dataset,Conditional_Dataset
from src.utils.render import model_Mesh_render


def path_process(opt):
    if_exists(opt.demo_input_dir)
    
    demo_result_dir=opt.demo_result_dir
    if opt.gt_demo:
        demo_path=os.path.join(demo_result_dir,'gt_demo')
    else:
        demo_path=os.path.join(demo_result_dir,'autoregressive_demo')

    models_path=os.path.join(demo_path,'predict_models')
    images_path=os.path.join(demo_path,'predict_images')
    objs_path=os.path.join(demo_path,'predict_objs')
    baseline_path=os.path.join(demo_path,'baseline')
    
    if_exists(opt.demo_result_dir)
    if_exists(demo_path)
    if_exists(models_path)
    if_exists(images_path)

    if opt.save_obj:
        if_exists(objs_path)
    if opt.baseline:
        if_exists(baseline_path)

    return models_path, images_path, objs_path, demo_path, baseline_path


def get_prompts(opt):
    prompts=[]
    prompt_files=os.listdir(opt.demo_input_dir)
    
    if prompt_files==[]:
        test_dataset=Simple_Dataset(opt,False)
        for path in test_dataset.paths:
            shutil.copy2(path, os.path.join(opt.demo_input_dir,path.split('/')[-1]))
        prompt_files=os.listdir(opt.demo_input_dir)
    
    for prompt_file in prompt_files:
        prompt_one=[]
        prompt_path=os.path.join(opt.demo_input_dir,prompt_file)
        with open(prompt_path,'r',encoding='utf-8') as prompt_input:
            try:
                prompt_lines=prompt_input.readlines()
            except:
                print(f'无法读取文件{prompt_path}')
            
            for line in prompt_lines:
                prompt_one.append(line.strip())
            
        prompts.append(prompt_one)

    result_names=[prompt[:-4] for prompt in prompt_files]

    return prompts, result_names
    

def render_demo(result,save_image_file,opt,save_obj_file=None):
    model = result['generation']
    model_render=model_Mesh_render(opt)
    
    if not opt.gt_demo:
        model.insert(0, '1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat')
    render_image=model_render.visualize_model(model,save_image_file,save_obj_file)


def main():
    opt=opts().return_args()
    models_path, images_path, objs_path, demo_path, baseline_path= path_process(opt)
    # new_bricks=os.listdir(opt.new_bricks_path)

    prompts, prompts_files = get_prompts(opt)
    # TODO
    # 如果是自回归的生成，可以直接用类别对应的数字做，那demo input等文件夹结构也就需要改改了
    texts = [name.split('_')[0] for name in prompts_files] if not opt.pretrain else None
    
    gt_prompts=prompts

    if not opt.gt_demo:
        prompts=[['1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat']]*len(prompts_files)

    generator, _ = Llama.build(
            ckpt_dir=opt.ckpt_dir,
            text_model_size=opt.text_model_size,
            max_seq_len=opt.max_seq_len,
            max_batch_size=opt.max_batch_size,
            position_dim=opt.position_dim,
            out_pad_dim=opt.out_pad_dim,
            rank=opt.rank,
            c_n_heads=opt.c_n_heads,
            text_dim=opt.text_dim,
            add_cross=opt.add_cross,
            seed=opt.seed,
            pretrain=opt.pretrain,
            )

    results = generator.lego_generation(
        prompts=prompts,
        texts=texts,
        max_gen_len=opt.max_gen_len,
        temperature=opt.temperature,
        top_p=opt.top_p,
        gt_demo=opt.gt_demo
    )
  
    torch.set_default_dtype(torch.float)

    batch_num=0
    model_files=[]
    for prompt, result in zip(gt_prompts, results):
        save_mpd_file=os.path.join(models_path,f'{prompts_files[batch_num]}.txt')
        save_image_file=os.path.join(images_path,f'{prompts_files[batch_num]}.png')
        
        with open(save_mpd_file,'w',encoding='utf-8') as file:
            if not opt.gt_demo:
                file.write('1 4 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat\n')
            for line in result['generation']:
                file.write(line+'\n')

        if opt.save_obj:
            save_obj_file=os.path.join(objs_path,f'{prompts_files[batch_num]}.obj')
            render_demo(result,save_image_file,opt,save_obj_file)
        else:
            render_demo(result,save_image_file,opt)
        
        batch_num+=1
        

if __name__ == "__main__":
    main()
