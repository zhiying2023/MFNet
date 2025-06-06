'''
功能：保存多个不同光场方法的每张图片的MAE到excle表格中，方便寻找与其他方法比较的可视化例子
推荐数据路径：
root_folder/
├── Method1/
│   ├── DUTLF-FS/
│   │   ├── img1.png
│   │   ├── img2.png
│   ├── HFUT/
│   │   ├── ...
│   ├── Lytro-Illum/
│       ├── ...
├── Method2/
│   ├── DUTLF-FS/
│   ├── HFUT/
│   ├── Lytro-Illum/
├── MethodN/
│   ├── ...
'''
import torch
import torch.nn.functional as F
import sys
import numpy as np
import os, argparse
import cv2
from Metric.saliency.eval_evaluator import Eval_thread
from Metric.saliency.eval_dataloader import EvalDataset
datasets = ['DUTLF-FS','HFUT','Lytro-Illum']
root_folder='./dataset/SOD_Map'
for method in os.listdir(root_folder):
    method_path = os.path.join(root_folder, method)
    # if method!='PICR-Net':
    #     continue
    for dataset in datasets:
        test_gt_root=os.path.join('/workspace/dataset/test',dataset,'test_masks')
        save_path=os.path.join(method_path,dataset)
        loader = EvalDataset(save_path, test_gt_root)
        #1.将结果保存到savepath
        #2.将每个数据集的每一张图片mae保存到路径(xlsxpath)，excle名字(method.xlsx)，子表名为各数据集(dataset)
        thread = Eval_thread(loader, 
                                savepath='/workspace/lfsod_savepath/log/log.txt',
                                xlsxpath='/workspace/test_best',
                                dataset=dataset, 
                                cuda=False,
                                method=method)
        print(print(thread.run()))
        print('{} {} Test Done!'.format(method,dataset))