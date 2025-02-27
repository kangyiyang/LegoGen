import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import shutil

working_path = os.getcwd()
import sys
sys.path.append(f'{working_path}/src')

from utils.utils import if_exists, get_paths


class Omr_Dataset_Preprocess():
    def __init__(self,
                 models_path,
                 source_bricks_path,
                 original_bricks_path,
                 preprocessed_models_path,
                 processed_models_path,
                 new_bricks_path,
                 max_s,
                 min_s,
                 good_rot
            ) -> None:
        """
        self.source_bricks:官方给定的砖块
        self.original_bricks_num:omr数据集中使用给定砖块的统计信息
        self.new_bricks_num:omr数据集中定义新砖块的统计信息
        self.bricks_num:omr数据集中所有砖块的统计信息
        self.original_bricks/self.new_bricks/self.bricks同理
        """
        self.models_path=models_path
        self.source_bricks_path=source_bricks_path
        self.original_bricks_path=original_bricks_path
        self.preprocessed_models_path=preprocessed_models_path
        self.processed_models_path=processed_models_path
        self.new_bricks_path=new_bricks_path

        self.source_bricks= os.listdir(fr'{self.source_bricks_path[0]}')
        self.source_other_bricks={}
        for i in range(len(self.source_bricks_path)-1):
            self.source_other_bricks[fr'{self.source_bricks_path[i+1]}']=os.listdir(fr'{self.source_bricks_path[i+1]}')

        self.original_bricks_num={}
        self.new_bricks_num={}
        self.bricks_num={}
        self.original_bricks=[]
        self.new_bricks=[]
        self.bricks=[]

        self.preprocessed_model=[]
        self.sub_modules=[]

        self.max_s=max_s
        self.min_s=min_s

        self.good_rot=good_rot

    def load_models_and_preprocess(self):
        model_paths=get_paths(self.models_path)

        for model_path in tqdm(model_paths,desc='Models Preprocessing'):
            with open(model_path,'r',encoding='utf-8') as file:
                try:
                    model=file.readlines()
                except:
                    print(f'无法读取文件{model_path}')
                # if model_path==r'':
                _ =self.model_preprocess(model)
                # 保存
                self.save_module(os.path.join(self.preprocessed_models_path,model_path.split('/')[-1]),self.preprocessed_model)
                self.preprocessed_model=[]
                self.sub_modules=[]
                    # break
    
    def models_process(self):
        model_paths=get_paths(self.preprocessed_models_path)
        rot_transfer=np.array([1,0,0,0,1,0,0,0,1.0])
        new_bricks = os.listdir(self.new_bricks_path)
        original_bricks = os.listdir(self.original_bricks_path)

        bricks_all=new_bricks+original_bricks
        bricks_new_name = {key: [dict({}),0] for key in bricks_all}

        for model_path in tqdm(model_paths,desc='Models process'):
            with open(model_path,'r',encoding='utf-8') as file:
                try:
                    model=file.readlines()
                except:
                    print(f'无法读取文件{model_path}')
                self.processed_model=[model[0].strip()]

                for line in model:
                    content=line.strip().split()
                    if not content:
                        continue
                    if content[0]=='1':
                        rot=' '.join(map(str, content[5:14]))
                        rot_array=np.array([float(x) for x in content[5:14]])
                        good_mse = np.mean((rot_array - self.good_rot) ** 2, axis=1)
                        good_line_rot = np.any(good_mse < 1e-2)
                        
                        if good_line_rot:
                            self.processed_model.append(line.strip())
                        else:
                            source_original=False
                            brick=' '.join(map(str, content[14:])).lower()
                            if '\\' in brick:
                                brick=brick.replace('\\', '_')
                            if brick in new_bricks:
                                brick_path=os.path.join(self.new_bricks_path,brick)
                            elif brick in original_bricks:
                                brick_path=os.path.join(self.original_bricks_path,brick)
                                source_original=True
                            else:
                                brick=brick.split('_')[1]
                                if brick in new_bricks:
                                    brick_path=os.path.join(self.new_bricks_path,brick)
                                else:
                                    print(f'未知砖块:{brick}')
                            
                            with open(brick_path,'r',encoding='utf-8') as brick_file:
                                try:
                                    brick_module=brick_file.readlines()
                                except:
                                    print(f'无法读取文件{brick_path}')
                            
                            content[5:14]=rot_transfer
                            new_brick=[brick[:-4],brick[-4:]]
                            assert brick[-4]=='.'

                            brick_rot_codename=bricks_new_name[brick][0]
                            brick_rots=list(brick_rot_codename.keys())
                            if len(brick_rots):
                                brick_rots_array=np.array([list(map(float, rot.split())) for rot in brick_rots])
                                rots_mse = np.mean((rot_array - brick_rots_array) ** 2, axis=1)
                                index = np.where(rots_mse < 1e-2)[0]
                            else:
                                index=[]

                            if len(index):
                                new_brick=new_brick[0]+f'_{index[0]+1}'+new_brick[1]
                            else:
                                bricks_new_name[brick][1]+=1
                                brick_rot_codename[rot]=bricks_new_name[brick][1]
                                new_brick=new_brick[0]+f'_{brick_rot_codename[rot]}'+new_brick[1]
                                new_module=[f'0 FILE {new_brick}\n',f'{line}\n','\n']+brick_module
                                new_brick_path=os.path.join(self.original_bricks_path,new_brick) if source_original else os.path.join(self.new_bricks_path,new_brick)
                                self.save_module(new_brick_path,new_module,if_module=True)

                            content_brick=content[:14]+[new_brick]
                            new_line=' '.join(map(str, content_brick))
                            self.processed_model.append(new_line)
                            
                self.save_module(os.path.join(self.processed_models_path,model_path.split('/')[-1]),self.processed_model)
        
        more_new_bricks = os.listdir(self.new_bricks_path)
        more_original_bricks = os.listdir(self.original_bricks_path)
        print(f'The original bricks used in omr dataset is {len(more_original_bricks)}')
        print(f'The new bricks used in omr dataset is {len(more_new_bricks)}')

    def model_preprocess(self,model,external=False):
        submodule_dependence=self.get_submodule_dependence(model)
        submodules=list(submodule_dependence.keys())
        submodules_split=[]
        for sub_module in submodules:
            submodules_split.append(sub_module.split())
        submodules=submodules+submodules_split
        
        start_line=np.min(np.array(list(submodule_dependence.values()))[:,0])
        for key,value in submodule_dependence.items():
            if start_line in value:
                main_module=model[submodule_dependence[key][0]:submodule_dependence[key][1]]
                break
        
        sub_line='1 16 0 0 0 1 0 0 0 1 0 0 0 1 3001.dat'
        self.preprocessed_model.append(model[0].strip())

        # if submodule_dependence[key][2]:
        #     self.preprocessed_model.append(f'1 16 0 0 0 1 0 0 0 1 0 0 0 1 {key}')
        #     path=os.path.join(self.new_bricks_path,key)
        #     self.save_module(path,model,if_module=True)
        #     print(key)
        # else:
        self.module_transfer(main_module,sub_line,submodule_dependence,submodules,model)

        for sub_module in self.sub_modules:
            for line in sub_module:
                if line[:5]!='0 BFC':
                    self.preprocessed_model.append(line.strip())
        
        if external:
            preprocessed_model=self.preprocessed_model
            self.preprocessed_model=[]
            return preprocessed_model

    # 返回一个model中所有的submodule及其对应的range,以及包括2，3，4，5线型与否
    def get_submodule_dependence(self,model):
        modules_lines={}
        modules=[]
        module_range=[]

        zero_parts=[]
        length=len(model)

        i=0
        while i<length:
            line=model[i]
            content=line.strip().split()
            if content and content[0]=='0':
                zero_parts.append(i)
                j=i+1
                while j and j<length:
                    line=model[j]
                    content=line.strip().split()
                    if content and content[0]!='0':
                        zero_parts.append(j)
                        break
                    elif content and content[0]=='0':
                        try:
                            if content[1]=='FILE':
                                zero_parts.append(j)
                                j-=1
                                break
                        except:
                            pass
                    j+=1
                i=j+1
            else:
                i+=1
        
        if len(zero_parts) % 2 != 0:
            zero_parts.append(len(model)-1)

        for i in range(int(len(zero_parts)/2)):
            start=zero_parts[2*i]
            end=zero_parts[2*i+1]

            names=[] 
            for j in range(start,end):
                line=model[j]
                content=line.strip()
                if content[2:6]=='FILE':
                    names.append(['FILE',content[7:].lower(),j])
                elif content[2:6]=='Name':
                    names.append(['Name',content[8:].lower(),j])
            
                for name in names:
                    if name[0]=='FILE':
                        modules.append(name[1])
                        module_range.append(name[2])
                        break
                    if name[0]=='Name':
                        modules.append(name[1])
                        module_range.append(name[2])
                        break

        for i in range(len(modules)-1):
            modules_lines[modules[i]]=[module_range[i],module_range[i+1],False]
            for line in model[module_range[i]:module_range[i+1]]:
                try:
                    line_type=line[0]
                    if line_type in ['2','3','4','5']:
                        modules_lines[modules[i]][2]=True
                        break
                except:
                    continue

                try:
                    content=line.strip().split()
                    brick=' '.join(map(str, content[14:])).lower()
                    if '\\' in brick:
                        brick=brick.split('\\')[1]
                    for key, value in self.source_other_bricks.items():
                        if brick in value:
                            modules_lines[modules[i]][2]=True
                            break
                    if modules_lines[modules[i]][2]:
                        break
                except:
                    pass

                try:
                    content=line.strip().split()
                    rot=np.array([float(x) for x in content[5:14]]).reshape(3,3)

                    # mse = np.mean((rot - self.gt_rot) ** 2, axis=1)
                    # good_line_rot = np.any(mse < 1e-3)
         
                    # if not good_line_rot:
                    #     modules_lines[modules[i]][2]=True
                    #     break

                    Q, R = np.linalg.qr(rot)
                    scale=np.abs(np.diag(R))
                    if np.max(scale)>self.max_s or np.min(scale)<self.min_s:
                        modules_lines[modules[i]][2]=True
                        break
                except:
                    continue

            # line_types=[item[0] for item in model[module_range[i]:module_range[i+1]]]
            # for line_type in line_types:
            #     if line_type in ['2','3','4','5']:
            #         modules_lines[modules[i]][2]=True
            #         break

        modules_lines[modules[-1]]=[module_range[-1],len(model),False]
        for line in model[module_range[-1]:]:
            try:
                line_type=line[0]
                if line_type in ['2','3','4','5']:
                    modules_lines[modules[-1]][2]=True
                    break
            except:
                continue

            try:
                content=line.strip().split()
                brick=' '.join(map(str, content[14:])).lower()
                if '\\' in brick:
                    brick=brick.split('\\')[1]
                for key, value in self.source_other_bricks.items():
                    if brick in value:
                        modules_lines[modules[-1]][2]=True
                        break
                if modules_lines[modules[-1]][2]:
                    break
            except:
                pass
            
            try:
                content=line.strip().split()
                rot=np.array([float(x) for x in content[5:14]]).reshape(3,3)

                # mse = np.mean((rot - self.gt_rot) ** 2, axis=1)
                # good_line_rot = np.any(mse < 1e-3)
        
                # if not good_line_rot:
                #     modules_lines[modules[i]][2]=True
                #     break
            
                Q, R = np.linalg.qr(rot)
                scale=np.abs(np.diag(R))
                if np.max(scale)>self.max_s or np.min(scale)<self.min_s:
                    modules_lines[modules[-1]][2]=True
                    break
            except:
                continue
        # line_types=[item[0] for item in model[module_range[-1]:]]
        # for line_type in line_types:
        #     if line_type in ['2','3','4','5']:
        #         modules_lines[modules[-1]][2]=True
        #         break
        
        return modules_lines

    def module_transfer(self,module,sub_line,submodule_dependence,submodules,model):
        for line in module:
            content=line.strip().split()
            if not content:
                continue
            if content[0]=='1':
                brick=' '.join(map(str, content[14:])).lower()
                # 普通砖块
                if brick in self.source_bricks:
                    new_line=self.line_transfer(line,sub_line.strip().split())
                    self.preprocessed_model.append(new_line)
                    self.brick_static_update(brick,self.original_bricks_num)
                    self.move_brick(brick,'original')
                # submodule
                elif brick in submodules or brick.split() in submodules:
                    # 新砖块
                    try:
                        temp=submodule_dependence[brick]
                    except:
                        for bad_module in list(submodule_dependence.keys()):
                            if brick.split()==bad_module.split():
                                brick=bad_module
                                break
                    sub_module=model[submodule_dependence[brick][0]:submodule_dependence[brick][1]]
                    if submodule_dependence[brick][2]:
                        if '\\' in brick:
                            brick=brick.replace('\\', '_')
                        new_line=self.line_transfer(line,sub_line.strip().split())
                        self.preprocessed_model.append(new_line)
                        self.brick_static_update(brick,self.new_bricks_num)
                        
                        self.already_exist=[brick]
                        self.new_sub_module=sub_module
                        self.new_sub_module_transfer(sub_module,submodule_dependence,submodules,model)
                        path=os.path.join(self.new_bricks_path,brick)
                        self.save_module(path,self.new_sub_module,if_module=True)
                    # 位置转换
                    else:
                        new_sub_line=self.line_transfer(line,sub_line.strip().split())
                        self.module_transfer(sub_module,new_sub_line,submodule_dependence,submodules,model)
                else:
                    if '\\' in brick:
                        brick=brick.split('\\')[1]
                    # 其余砖块
                    source_brick_path=None
                    for key, value in self.source_other_bricks.items():
                        if brick in value:
                            source_brick_path = key
                            self.brick_static_update(brick,self.new_bricks_num)
                            self.move_brick(brick,'new',source_brick_path)
                            break
                    # 未知砖块
                    if not source_brick_path:
                        print(f"未知砖块:{brick}")
                    new_line=self.line_transfer(line,sub_line.strip().split())
                    self.preprocessed_model.append(new_line)

    def line_transfer(self,line,sub_line):
        content=line.strip().split()
        sub_position=sub_line[2:14]

        # position transfer
        old_trans=list(map(lambda x: float(x), content[2:5]))
        old_rot=list(map(lambda x: float(x), content[5:14]))
        sub_trans=list(map(lambda x: float(x), sub_position[0:3]))
        sub_rot=list(map(lambda x: float(x), sub_position[3:12]))
        
        sub_pose=np.zeros((4,4))
        sub_pose[:3,:3]=np.array(sub_rot).reshape((3,3))
        sub_pose[:3,3]=np.array(sub_trans)
        sub_pose[3,3]=1
        

        old_pose=np.zeros((4,4))
        old_pose[:3,:3]=np.array(old_rot).reshape((3,3))
        old_pose[:3,3]=np.array(old_trans)
        old_pose[3,3]=1

        new_pose=sub_pose.dot(old_pose)
        content[2:5]=new_pose[:3,3]
        content[5:14]=new_pose[:3,:3].reshape(-1)

        # color transfer
        old_color=content[1]
        sub_color=sub_line[1]
        content[1]=sub_color if old_color=='16' else old_color
        
        new_line=' '.join(map(str, content))
        
        return new_line

    def new_sub_module_transfer(self,new_module,submodule_dependence,submodules,model):
        for line in new_module:
            content=line.strip().split()
            if not content:
                continue
            if content[0]=='1':
                brick=' '.join(map(str, content[14:])).lower()
                if brick in submodules or brick.split() in submodules:
                    try:
                        temp=submodule_dependence[brick]
                    except:
                        for bad_module in list(submodule_dependence.keys()):
                            if brick.split()==bad_module.split():
                                brick=bad_module
                                break
                    sub_module=model[submodule_dependence[brick][0]:submodule_dependence[brick][1]]
                    if brick not in self.already_exist:
                        self.already_exist.append(brick)
                        self.new_sub_module+=sub_module
                        self.new_sub_module_transfer(sub_module,submodule_dependence,submodules,model)

    def brick_static_update(self,brick,bricks_num):
        if brick in bricks_num.keys():
            bricks_num[brick]+=1
        else:
            bricks_num[brick]=1

    def save_module(self,path,module,if_module=False):
        with open(path,'w',encoding='utf-8') as file:
            for line in module:
                try:
                    file.write(line)
                    if not if_module:
                        file.write('\n')
                except:
                    print(f'无法保存文件')

    def brick_static_visual(self):
        original_bricks_num_sorted=dict(sorted(self.original_bricks_num.items(),key=lambda x:x[1],reverse=True))
        self.original_bricks=original_bricks_num_sorted.keys()
        num_original_sorted=list(original_bricks_num_sorted.values())
        new_bricks_num_sorted=dict(sorted(self.new_bricks_num.items(),key=lambda x:x[1],reverse=True))
        self.new_bricks=list(new_bricks_num_sorted.keys())
        num_new_sorted=list(new_bricks_num_sorted.values())

        #统计砖块数量
        x_brick=np.arange(len(self.original_bricks))+1
        print(f'The original bricks used in omr dataset is {len(self.original_bricks)}')
        plt.loglog(x_brick,num_original_sorted)
        plt.xlabel('brick_id')  # 横轴标签
        plt.ylabel('occurence')  # 纵轴标签
        plt.savefig('original_bricks.png')
        plt.clf()

        x_brick=np.arange(len(self.new_bricks))+1
        print(f'The new bricks used in omr dataset is {len(self.new_bricks)}')
        plt.loglog(x_brick,num_new_sorted)
        plt.xlabel('brick_id')  # 横轴标签
        plt.ylabel('occurence')  # 纵轴标签
        plt.savefig('new_bricks.png')
        
    def move_brick(self,brick,type,source_brick_path='new brick path'):
        if type=='original':
            source_file = fr'{self.source_bricks_path[0]}/{brick}'
            destination_folder = fr'{self.original_bricks_path}/{brick}'
            shutil.copy2(source_file, destination_folder)
        elif type=='new':
            source_file = fr'{source_brick_path}/{brick}'
            destination_folder = fr'{self.new_bricks_path}/{brick}'
            shutil.copy2(source_file, destination_folder)


models_path=os.path.join(working_path,'dataset/pretrain_bricks dataset/models')
source_bricks_path=[]
parts_path=['parts','parts/s','p','p/8','p/48']
for part_path in parts_path:
    source_bricks_path.append(os.path.join(working_path,f'dataset/complete/{part_path}'))

preprocessed_models_path=os.path.join(working_path,'dataset/pretrain_bricks dataset/preprocessed_models')
processed_models_path=os.path.join(working_path,'dataset/pretrain_bricks dataset/processed_models')
original_bricks_path=os.path.join(working_path,'dataset/bricks/original_bricks')
new_bricks_path=os.path.join(working_path,'dataset/bricks/new_bricks')

max_s=1.05
min_s=0.95

good_rot=np.array(
            [
                [1,0,0, 0,1,0, 0,0,1],
                [1,0,0, 0,0,1, 0,-1,0],
                [1,0,0, 0,-1,0, 0,0,-1],
                [1,0,0, 0,0,-1, 0,1,0],

                [0,1,0, -1,0,0, 0,0,1],
                [0,1,0, 0,0,-1, -1,0,0],
                [0,1,0, 1,0,0, 0,0,-1],
                [0,1,0, 0,0,1, 1,0,0],

                [-1,0,0, 0,-1,0, 0,0,1],
                [-1,0,0, 0,0,1, 0,1,0],
                [-1,0,0, 0,1,0, 0,0,-1],
                [-1,0,0, 0,0,-1, 0,-1,0],

                [0,-1,0, 1,0,0, 0,0,1],
                [0,-1,0, 0,0,1, -1,0,0],
                [0,-1,0, -1,0,0, 0,0,-1],
                [0,-1,0, 0,0,-1, 1,0,0],

                [0,0,1, 1,0,0, 0,1,0],
                [0,0,1, 0,1,0, -1,0,0],
                [0,0,1, -1,0,0, 0,-1,0],
                [0,0,1, 0,-1,0, 1,0,0],

                [0,0,-1, 1,0,0, 0,-1,0],
                [0,0,-1, 0,1,0, 1,0,0],
                [0,0,-1, -1,0,0, 0,1,0],
                [0,0,-1, 0,-1,0, -1,0,0],
            ]
        )

Model_analysis=Omr_Dataset_Preprocess(models_path,
                                      source_bricks_path,
                                      original_bricks_path,
                                      preprocessed_models_path,
                                      processed_models_path,
                                      new_bricks_path,
                                      max_s,
                                      min_s,
                                      good_rot)


if __name__ == '__main__':
    if_exists(preprocessed_models_path)
    if_exists(processed_models_path)
    if_exists(original_bricks_path)
    if_exists(new_bricks_path)
    Model_analysis.load_models_and_preprocess()
    Model_analysis.brick_static_visual()
    # Model_analysis.models_process()