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
from src.utils.utils import if_exists,images_transform
from src.dataset import OMR_Pretrain_Dataset,Conditional_Dataset
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
    
    if_exists(opt.demo_result_dir)
    if_exists(demo_path)
    if_exists(models_path)
    if_exists(images_path)
    if opt.save_obj:
        if_exists(objs_path)

    return models_path, images_path, objs_path, demo_path


def get_prompts(opt):
    prompts=[]
    prompt_files=os.listdir(opt.demo_input_dir)
    
    if prompt_files==[]:
        test_dataset=OMR_Pretrain_Dataset(opt,demo=True)
        for path in test_dataset.test_paths:
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
            
            for line in prompt_lines[1:]:
                prompt_one.append(line.strip())
            
        prompts.append(prompt_one)

    result_names=[prompt[:-4] for prompt in prompt_files]

    return prompts, result_names

def get_con_prompts(opt):
    prompts=[]
    images=[]

    models_path=os.path.join(opt.demo_input_dir,'models')
    images_path=os.path.join(opt.demo_input_dir,'images')
    if_exists(models_path)
    if_exists(images_path)

    prompt_files=os.listdir(models_path)
    image_files=os.listdir(images_path)
    
    if prompt_files==[]:
        test_dataset=Conditional_Dataset(opt,'test')
        source_models=test_dataset.model_files
        source_images=test_dataset.image_files
        for i in range(len(source_models)):
            shutil.copy2(source_models[i], os.path.join(models_path,source_models[i].split('/')[-1]))
            shutil.copy2(source_images[i], os.path.join(images_path,source_images[i].split('/')[-1]))
        
        prompt_files=os.listdir(models_path)
        image_files=os.listdir(images_path)
    
    result_names=[prompt[:-4] for prompt in prompt_files]

    for prompt_file in prompt_files:
        prompt_one=[]
        prompt_path=os.path.join(models_path,prompt_file)
        with open(prompt_path,'r',encoding='utf-8') as prompt_input:
            try:
                prompt_lines=prompt_input.readlines()
            except:
                print(f'无法读取文件{prompt_path}')
            
            for line in prompt_lines[1:]:
                prompt_one.append(line.strip())
            
        prompts.append(prompt_one)
    
    for image_file in image_files:
        image_path=os.path.join(images_path,image_file)
        image = Image.open(image_path).convert('RGB')
        images.append(image)

    return prompts, images, result_names

def caculate_demo_acc(prompt,result,opt):
    generation_result=result['generation']
    prompt_seq_len=len(prompt)
    result_seq_len=len(generation_result)
    
    if prompt_seq_len>result_seq_len:
        prompt=prompt[:result_seq_len]
    else:
        generation_result=generation_result[:prompt_seq_len]

    seq_len=min(prompt_seq_len,result_seq_len)

    prompt_bricks=[' '.join(map(str, line.strip().split()[14:])).lower() for line in prompt]
    result_bricks=[' '.join(map(str, line.strip().split()[14:])).lower() for line in generation_result]
    count=0
    for prompt_brick,result_brick in zip(prompt_bricks,result_bricks):
        if prompt_brick==result_brick:
            count+=1
    brick_acc=count/seq_len

    prompt_trans=torch.tensor([[float(x) for x in line.strip().split()[2:5]] for line in prompt])
    result_trans=torch.tensor([[float(x) for x in line.strip().split()[2:5]] for line in generation_result])
    trans_mse = F.mse_loss(
                    input=prompt_trans,
                    target=result_trans,
                    reduction="none",
                )

    count = torch.sum(trans_mse < opt.threshold)
    trans_acc = count / torch.numel(trans_mse)

    prompt_rot=torch.tensor([[float(x) for x in line.strip().split()[5:14]] for line in prompt]).reshape(seq_len,3,3)
    result_rot=torch.tensor([[float(x) for x in line.strip().split()[5:14]] for line in generation_result]).reshape(seq_len,3,3)
    rot_similarity=torch.mean((F.cosine_similarity(prompt_rot,result_rot,dim=2)+1)/2)
    
    return brick_acc,trans_acc,rot_similarity


def ldraw_process(result, save_mpd_file, opt, new_bricks, name):
    add_bricks=[]
    file=open(save_mpd_file,'w',encoding='utf-8')
    file.write(f'0 FILE {name}.ldr\n')

    for line in result['generation']:
        content=line.strip().split()
        brick=' '.join(map(str, content[14:])).lower()
        if brick in new_bricks:
            add_bricks.append(brick)
        if '_' in brick:
            line=line.replace('_', '\\')   
        file.write(line+'\n')

    add_bricks = list(set(add_bricks))
    for add_brick in add_bricks:
        file.write('\n')
        brick_file=os.path.join(opt.new_bricks_path,add_brick)
        with open(brick_file,'r',encoding='utf-8') as add_brick_file:
            lines=add_brick_file.readlines()
        
        for line in lines:
            content=line.strip().split()
            try:
                if not content or (content[0]=='0' and content[1]=='BFC'):
                    continue
            except:
                continue
            file.write(line)

    file.close()

def render_demo(result,save_image_file,opt,save_obj_file=None):
    model = result['generation']
    model_render=model_Mesh_render(opt)
    render_image=model_render.visualize_model(model,save_image_file,save_obj_file)


def main():
    opt=opts().return_args()
    models_path, images_path, objs_path, demo_path = path_process(opt)
    new_bricks=os.listdir(opt.new_bricks_path)

    if opt.pretrain:
        prompts, prompts_files = get_prompts(opt)
        images = None
    else:
        prompts, images, prompts_files = get_con_prompts(opt)
    
    gt_prompts=prompts
    if opt.gt_demo:
        max_length = max(len(lst) for lst in prompts)+2
    else:
        prompts=[[]]*len(prompts_files)
        max_length = opt.max_seq_len

    generator, _ = Llama.build(
            ckpt_dir=opt.ckpt_dir,
            image_model_dir=opt.image_model_dir,
            image_model_size=opt.image_model_size,
            original_bricks_path=opt.original_bricks_path,
            new_bricks_path=opt.new_bricks_path,
            max_seq_len=max_length,
            max_batch_size=opt.max_batch_size,
            position_dim=opt.position_dim,
            out_pad_dim=opt.out_pad_dim,
            # out_rot_dim=opt.out_rot_dim,
            rank=opt.rank,
            c_n_heads=opt.c_n_heads,
            patch_h=opt.patch_h,
            patch_w=opt.patch_w,
            image_dim=opt.image_dim,
            add_cross=opt.add_cross,
            seed=opt.seed,
            pretrain=opt.pretrain,
            )


    results = generator.lego_generation(
        opt=opt,
        prompts=prompts,
        images=images,
        max_gen_len=opt.max_gen_len,
        temperature=opt.temperature,
        top_p=opt.top_p,
        gt_demo=opt.gt_demo
    )
  
    torch.set_default_dtype(torch.float)

    batch_num=0
    brick_acc_all=[]
    rot_similarity_all=[]
    trans_acc_all=[]

    for prompt, result in zip(gt_prompts, results):
        if opt.demo_acc:
            brick_acc,trans_acc,rot_similarity=caculate_demo_acc(prompt,result,opt)
            brick_acc_all.append(brick_acc)
            trans_acc_all.append(trans_acc)
            rot_similarity_all.append(rot_similarity)

        save_mpd_file=os.path.join(models_path,f'{prompts_files[batch_num]}.txt')
        save_image_file=os.path.join(images_path,f'{prompts_files[batch_num]}.png')
        
        ldraw_process(result,save_mpd_file,opt,new_bricks,prompts_files[batch_num])
        if opt.save_obj:
            save_obj_file=os.path.join(objs_path,f'{prompts_files[batch_num]}.obj')
            render_demo(result,save_image_file,opt,save_obj_file)
        else:
            render_demo(result,save_image_file,opt)
        
        batch_num+=1
    
    brick_acc_mean=np.mean(brick_acc_all)
    trans_acc_mean=torch.mean(torch.stack(trans_acc_all))
    rot_similarity_mean=torch.mean(torch.stack(rot_similarity_all))
    acc=opt.alpha*brick_acc_mean+opt.gamma*trans_acc_mean+opt.delta*rot_similarity_mean

    message=f'brick_acc: {brick_acc_mean} trans_acc: {trans_acc_mean} rot_similarity: {rot_similarity_mean} acc: {acc}'
    print(message)

    if opt.save_demo_acc:
        with open(os.path.join(demo_path,'acc.txt'),'w',encoding='utf-8') as file:
            file.write(message)
        

if __name__ == "__main__":
    main()
