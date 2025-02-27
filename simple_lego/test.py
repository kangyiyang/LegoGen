import os
import pickle
import torch
import json
from src.utils.options import opts
from tqdm import tqdm



names_to_id={
    'car':0,
    'chair':1,
    'couch':2,
    'cup':3,
    'flat-block':4,
    'hollow-cylinder':5,
    'line':6,
    'pyramid':7,
    'table':8,
    'tall-block':9,
    'tower':10,
    'wall':11,
}


def caculate_demo_acc(model,classes,config,graphs=None,run_dir=None):
    evaluator = dgmg_helpers.LegoModelEvaluation(v_max = config.max_generated_graph_size,edge_max = config.max_edges_per_node)
    
    model.eval()
    with torch.no_grad():    
        gin_metrics, lego_metrics = evaluator.test_model(model,
                                                        sampled_graphs=graphs, 
                                                        classes_to_generate=classes*5, 
                                                        num_samples = len(classes)*5, 
                                                        dir = run_dir)
        

        message='own: '  if run_dir is None else 'baseline-valid: '
        for key,value in gin_metrics.items():
            message += '%s: %.7f ' % (key, value)        
        print(message)

        for key,value in lego_metrics.items():
            print(key,value)
        
        
        if run_dir is not None:
            save_results(run_dir, lego_metrics, gin_metrics, 1)
    
    return message



def save_results(run_dir, lego_metrics, gin_metrics, epoch):
        with open(os.path.join(run_dir, 'results.h5'), 'rb') as f:
            prev_results = pickle.load(f)
        for key, val in lego_metrics.items():
            gin_metrics[key] = val
        prev_results[epoch] = gin_metrics
        with open(os.path.join(run_dir, 'results_temp.h5'), 'wb') as f:
            pickle.dump(prev_results, f)
        os.system('mv {}/results_temp.h5 {}/results.h5'.format(run_dir, run_dir))


def test_model(opt):
    opt.num_shifts = 7
    opt.num_node_types = 2

    opt.auto_implied_edges = True
    opt.force_valid = True

    file = torch.load(opt.baseline_model)
    model = DGMG(**vars(opt))
    model.load_state_dict(file['model_state_dict'])
    
    demo='autoregressive_demo' if not opt.gt_demo else 'gt_demo'
    demo_path=os.path.join(opt.test_input_dir,demo)
    run_dir=os.path.join(demo_path,'baseline')
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    prompt_files=os.listdir(os.path.join(demo_path,'predict_models'))
    names=[prompt[:-4].split('_')[0] for prompt in prompt_files]
    classes=[names_to_id[name] for name in names]

    #-------
    graphs=[]
    for prompt_file in prompt_files:
        model_file=os.path.join(demo_path,f'predict_models/{prompt_file}')
        graphs.append(LDraw.LDraw_to_graph(model_file))
    
    acc_message=caculate_demo_acc(model,classes,opt,graphs=graphs)
    with open(os.path.join(demo_path,'acc.txt'),'w',encoding='utf-8') as file:
        file.write(acc_message+'\n')
    
    if opt.baseline:
        # The file to store results
        with open(os.path.join(run_dir, 'results.h5'), 'wb') as f:
            pickle.dump({}, f)
        
        baseline_acc_message=caculate_demo_acc(model,classes,opt,run_dir=run_dir)
        with open(os.path.join(demo_path,'acc.txt'),'a',encoding='utf-8') as file:
            file.write(baseline_acc_message+'\n')


def baseline_render(opt):
    demo='autoregressive_demo' if not opt.gt_demo else 'gt_demo'
    demo_path=os.path.join(opt.test_input_dir,demo)
    run_dir=os.path.join(demo_path,'baseline')

    model_render=model_Mesh_render(opt)
    ldr_path=os.path.join(run_dir,'ldr_files')
    save_image_path=os.path.join(run_dir,'images')
    if not os.path.exists(save_image_path):
        os.makedirs(save_image_path)
    ldr_paths=get_paths(ldr_path)

    for path in tqdm(ldr_paths):
        new_model=[]
        with open(path, 'r', encoding='utf-8') as file:
            model=file.readlines()
            for line in model:
                content=line.strip().split()
                content[1]='4'
                new_line=' '.join(map(str, content))
                new_model.append(new_line)
            name=path.split('/')[-1][:-4]
            save_image_file=os.path.join(save_image_path, f'{name}.png')
            render_image=model_render.visualize_model(new_model,save_image_file)


def best_baseline(opt):
    import pickle

    pickle_file = '/NASdata/yyk/lego-DGMG/15-05-2024--13-37-16/results.h5'
    with open(pickle_file, 'rb') as f:
        dataset_with_filenames = pickle.load(f)


    fid,kid,Gin_acc,p,r,d,c,v=[],[],[],[],[],[],[],[]

    for key, value in dataset_with_filenames.items():
        if key>=180 and value['kid']>0:
            fid.append(value['fid'])
            kid.append(value['kid'])
            Gin_acc.append(value['GIN_accuracy'])
            p.append(value['precision'])
            r.append(value['recall'])
            d.append(value['density'])
            c.append(value['coverage'])
            v.append(value['Overall invalid lego build (%)'])


    message=f'baseline: fid:{sum(fid)/len(fid)}  kid:{sum(kid)/len(kid)} Gin_acc:{sum(Gin_acc)/len(Gin_acc)} p:{sum(p)/len(p)} r:{sum(r)/len(r)} d:{sum(d)/len(d)} c:{sum(c)/len(c)} v:{100-sum(v)/len(v)}'
    print(message)
    demo='autoregressive_demo' if not opt.gt_demo else 'gt_demo'
    demo_path=os.path.join(opt.test_input_dir,demo)
    with open(os.path.join(demo_path,'acc.txt'),'a',encoding='utf-8') as file:
        file.write(message)




if __name__ == '__main__':
    opt=opts().return_args()
    if opt.baseline_render:
        from src.utils.utils import get_paths
        from src.utils.render import model_Mesh_render
        baseline_render(opt)
    else:
        from GenerativeLEGO.pyFiles import LDraw
        from GenerativeLEGO.DGL_DGMG.model_batch import DGMG
        from GenerativeLEGO.DGL_DGMG import dgmg_helpers
        test_model(opt)
        best_baseline(opt)
