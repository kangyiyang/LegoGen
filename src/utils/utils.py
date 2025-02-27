import os
import torch
import numpy as np
from typing import List
import torch.nn.functional as F
import torchvision.transforms as T

from fairscale.nn.model_parallel.initialize import model_parallel_is_initialized

#---------------------------------------------------------------
#---------------------------------------------------------------
# os utils
def get_paths(path)->List[str]:
    paths=[]
    files = os.listdir(path)
    for file in files:
        paths.append(os.path.join(path,file))

    return paths

def if_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

#---------------------------------------------------------------
#---------------------------------------------------------------
# gpu utils
# def get_default_device() -> torch.device:
#     if torch.cuda.is_available() and torch.cuda.is_initialized() and opt.cuda:
#         return torch.device("cuda")
#     else:
#         return torch.device("cpu")

# def is_distributed() -> bool:
#     return False
        
#---------------------------------------------------------------
#---------------------------------------------------------------
# pretrain_postprocess utils
def rotation_6d_to_9d(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    b=torch.stack((b1, b2, b3), dim=-2)
    return torch.flatten(b, start_dim=2)

def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def seq_sample(probs, p):
    """
    对每个 batch 进行 multinomial 采样的函数。
    """

    batch_size, seq_len, class_num = probs.size()

    # 创建结果张量
    sampled_tokens = torch.zeros(batch_size, seq_len, dtype=torch.long)

    # 对每个 batch 进行循环采样
    for i in range(batch_size):
        # 从概率分布中采样多个标记
        prob=probs[i]
        sampled_tokens[i] = sample_top_p(prob,p).reshape(-1)

    return sampled_tokens


#---------------------------------------------------------------
#---------------------------------------------------------------
# metric utils
def batch_qr_decomposition(gt_r_s):
    """
    执行批量矩阵的 QR 分解

    参数:
        A (torch.Tensor): 形状为 [batch_size, 3, 3] 的批量矩阵

    返回:
        Q (torch.Tensor): 正交矩阵，形状为 [batch_size, 3, 3]
        R (torch.Tensor): 上三角矩阵，形状为 [batch_size, 3, 3]
    """
    batch_seq = gt_r_s.size(0)
    float_gt_r_s = gt_r_s.float()

    # 初始化空的 Q 和 R 列表
    Q_list = []
    R_list = []

    # 遍历批量矩阵并执行 QR 分解
    for i in range(batch_seq):  # 获取当前批量中的矩阵并重塑形状为 3x3
        Q_i, R_i = torch.linalg.qr(float_gt_r_s[i])  # 执行 QR 分解
        Q_list.append(Q_i)
        R_list.append(R_i.diag())

    # 将结果转换为张量
    Q = torch.stack(Q_list)
    R = torch.stack(R_list)
    del float_gt_r_s

    return Q, R

def compute_box_vertices(center,axes,scale=1):
    # 将输入的列表转换为PyTorch张量
    center = center.to(torch.float32)
    axes = axes.to(torch.float32).reshape(3, 3)*scale

    # 计算顶点相对中心点的偏移量
    offsets = torch.tensor([[0, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [1, 1, 0],
                            [1, 0, 1],
                            [0, 1, 1],
                            [1, 1, 1]],dtype=torch.float32)
    # offsets = torch.tensor([[-0.5, -0.5, -0.5],
    #                          [-0.5, -0.5, 0.5],
    #                          [-0.5, 0.5, -0.5],
    #                          [-0.5, 0.5, 0.5],
    #                          [0.5, -0.5, -0.5],
    #                          [0.5, -0.5, 0.5],
    #                          [0.5, 0.5, -0.5],
    #                          [0.5, 0.5, 0.5]],dtype=torch.float32)

    # 计算顶点的位置
    vertices = center + torch.matmul(axes, offsets.T).T

    return vertices

def calculate_iou(box1, box2):
    # 计算第一个长方体的最小和最大坐标
    min_x_box1 = min(box1[:, 0])
    min_y_box1 = min(box1[:, 1])
    min_z_box1 = min(box1[:, 2])
    max_x_box1 = max(box1[:, 0])
    max_y_box1 = max(box1[:, 1])
    max_z_box1 = max(box1[:, 2])

    # 计算第二个长方体的最小和最大坐标
    min_x_box2 = min(box2[:, 0])
    min_y_box2 = min(box2[:, 1])
    min_z_box2 = min(box2[:, 2])
    max_x_box2 = max(box2[:, 0])
    max_y_box2 = max(box2[:, 1])
    max_z_box2 = max(box2[:, 2])

    # 计算交集体积
    overlap_length_x = max(0, min(max_x_box1, max_x_box2) - max(min_x_box1, min_x_box2))
    overlap_length_y = max(0, min(max_y_box1, max_y_box2) - max(min_y_box1, min_y_box2))
    overlap_length_z = max(0, min(max_z_box1, max_z_box2) - max(min_z_box1, min_z_box2))
    intersection_volume = overlap_length_x * overlap_length_y * overlap_length_z

    # 计算并集体积
    volume_box1 = (max_x_box1 - min_x_box1) * (max_y_box1 - min_y_box1) * (max_z_box1 - min_z_box1)
    volume_box2 = (max_x_box2 - min_x_box2) * (max_y_box2 - min_y_box2) * (max_z_box2 - min_z_box2)
    union_volume = volume_box1 + volume_box2 - intersection_volume

    # 计算3D IoU
    iou = intersection_volume / (union_volume + 1e-6)

    return iou

#---------------------------------------------------------------
#---------------------------------------------------------------
# loss utils
def focal_loss(ce_loss, alpha=0.25, gamma=2):
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss
    return torch.mean(focal_loss)


def compute_angles(matrix1, matrix2):
    """
    计算两个 3x3 矩阵中三个轴之间的夹角（以弧度为单位）。
    Args:
        matrix1 (torch.Tensor): 第一个矩阵。
        matrix2 (torch.Tensor): 第二个矩阵。
    Returns:
        torch.Tensor: 三个轴之间的夹角（以弧度为单位）。
    """
    # 提取矩阵的三个轴向量
    vectors1 = matrix1.unbind(dim=1)
    vectors2 = matrix2.unbind(dim=1)

    angles = []
    for i in range(3):
        if torch.all(vectors1[i]==0):
            angle = 0
        else:
            angle = torch.acos(torch.dot(vectors1[i], vectors2[i]) / (torch.norm(vectors1[i]) * torch.norm(vectors2[i])))
        angles.append(angle)

    return torch.mean(torch.stack(angles))

def rot_angle_loss(input,target):
    batch_seq, _, _=input.shape
    angle_loss=[]

    for i in range(batch_seq):
        angle_loss.append(compute_angles(input[i],target[i]))
    
    return torch.mean(torch.stack(angle_loss))

def scale_magnification_loss(input,target,reduction,pad_position):
    scale_loss_init = target / (input + 1e-6)
    #---------------------------------------------
    # 这个操作是不可微的，确实哦，怪不得不能作为损失
    non_zero_rows = torch.where(~torch.all(target == pad_position, dim=1))[0]
    # 根据非零行的索引，创建新的矩阵
    non_zero_loss = scale_loss_init[non_zero_rows]
    #---------------------------------------------

    new_scale_loss = torch.where((non_zero_loss > -1) & (non_zero_loss < 1), 1 / (non_zero_loss+1e-6), non_zero_loss)
    scale_metric=torch.ones_like(non_zero_loss)
    scale_loss = F.mse_loss(
                input=new_scale_loss,
                target=scale_metric,
                reduction=reduction,
                )

    return scale_loss

#---------------------------------------------------------------
#---------------------------------------------------------------
# color utils
def hex_to_rgb(hex_code,alpha=1):
    # 去除可能存在的 '#' 字符
    hex_code = hex_code.strip('#')
    
    # 将16进制代码转换为RGB值
    r = int(hex_code[0:2], 16)
    g = int(hex_code[2:4], 16)
    b = int(hex_code[4:6], 16)
    
    rgb=np.array([r,g,b])
    if alpha!=1:
        rgb=np.int32(rgb+(1-alpha)*(256-rgb))

    # 返回RGB颜色值
    return rgb

def color_config(color_config_path):
    colors={}
    with open(color_config_path,'r',encoding='utf-8') as file:
        config=file.readlines()
        for line in config:
            if 'CODE' in line:
                content=line.strip().split()
                color_id=content[content.index('CODE')+1]
                color_hex=content[content.index('VALUE')+1]
                egde_color_hex=content[content.index('EDGE')+1]
                alpha=int(content[content.index('ALPHA')+1])/256 if 'ALPHA' in line else 1
                
                colors[color_id]={
                    'color':hex_to_rgb(color_hex,alpha),
                    'egde_color':hex_to_rgb(egde_color_hex,alpha)
                    }
    
    return colors

#---------------------------------------------------------------
#---------------------------------------------------------------
# data utils
def multi_seq(paths,opt):
    max_seq_len=opt.max_seq_len-2
    models=[]

    for index in range(len(paths)):
        with open(paths[index],'r',encoding='utf-8') as file:            
            model=file.readlines()[1:]
            seq_len=len(model)

            if seq_len<=max_seq_len:
                models.append([model,1,1])
            else:
                multi_num=int(seq_len/max_seq_len)+1
                start=0
                end=max_seq_len
                if_bos=1
                if_eos=0

                while multi_num-1:
                    if model[start:end]==[]:
                        break
                    models.append([model[start:end],if_bos,if_eos])
                    if_bos=0
                    start=end
                    end+=max_seq_len
                    multi_num-=1
                
                if model[start:end]!=[]:
                    models.append([model[start:end],0,1])

    return models

def images_transform(image,opt):    
    transform = T.Compose([
        T.GaussianBlur(9, sigma=(0.1, 2.0)),
        T.Resize((opt.patch_h * 14, opt.patch_w * 14)),
        # T.CenterCrop((opt.patch_h * 14, opt.patch_w * 14)),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    return transform(image)
