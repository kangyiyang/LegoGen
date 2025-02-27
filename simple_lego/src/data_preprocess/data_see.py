import os
import numpy as np

working_path = os.getcwd()
import sys
sys.path.append(f'{working_path}/src')

from utils.utils import if_exists, get_paths
from typing import List, Dict, Literal, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib

# 设置中文字体，这里以SimHei（黑体）为例，你需要确保你的系统中已安装该字体或者替换为你系统中存在的中文字体路径
plt.rcParams['font.sans-serif'] = ['SimHei']
# 设置正常显示符号
matplotlib.rcParams['axes.unicode_minus'] = False 

def plot_histogram(data_distribution):
    """
    绘制数据分布柱状图。
    
    参数:
    data_distribution (dict): 一个字典，键为数据范围的字符串表示，值为该范围内数据的数量。
    
    """

    plt.rcParams['font.size'] = 16

    # 数据准备
    bins = list(data_distribution.keys())
    counts = list(np.array(list(data_distribution.values()))/3.6)
    
    # 颜色列表，用于不同区间（尽管当前示例只有一种情况，但保留扩展性）
    colors = ['#607D8B']
    
    # 创建柱状图
    plt.figure(figsize=(10, 6))  # 可调整图表大小
    bars = plt.bar(bins, counts, color=colors[:len(counts)])  # 确保颜色与条形数量匹配

    # # 添加图例 - 即使只有一个系列，也可以象征性地添加
    # plt.legend(handles=bars, labels=bins, title="Value Ranges")  # 使用区间作为图例项

    # 添加标题和标签
    plt.title('Com数据集序列长度分布')
    plt.xlabel('值分布')
    plt.ylabel('比例/%')

    # 显示网格
    plt.grid(axis='y', linestyle='--', linewidth=0.7, alpha=0.7)

    # 自动调整x轴刻度标签以避免重叠
    plt.xticks(rotation=45, ha='right')  # 旋转并右对齐标签，以改善可读性

    # 显示图形
    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    plt.savefig('/home/yyk/simple_lego/dataset/brick_static/test.png')


models_path=os.path.join(working_path,'dataset/preprocessed_dataset')
ldraw_paths=get_paths(models_path)

length_static=[20,40,60,80,100]
num_length=[0,0,0,0,0,0]

for path in ldraw_paths:
    with open(path,mode='r',encoding='utf-8') as ldraw_file:
        model=ldraw_file.readlines()
        length=len(model)
        for i in range(len(length_static)):
            if length<length_static[i]:
                num_length[i]+=1
                break
        if length>length_static[-1]:
            num_length[-1]+=1

num_dict={
    '<20':num_length[0],
    '20-40':num_length[1],
    '40-60':num_length[2],
    '60-80':num_length[3],
    '80-100':num_length[4],
    '>100':num_length[5],
    }

plot_histogram(num_dict)