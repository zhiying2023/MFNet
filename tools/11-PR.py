'''
功能：画PR曲线
font_path：中文字体SimHei.ttf路径
gt_root：GT路径
pred_root：预测的显著性图路径
datasets：比较PR曲线数据集
colors：每个方法对应PR曲线颜色
method：PR曲线比较的方法，这个也可以用os.listdir获取，我这里用列表是想让我的方法呈现红色
savepath:PR图保存路径
推荐路径：
gt_root
├── DUTLF-FS/
│   ├── img1.png
│   ├── img2.png
├── HFUT/
│   ├── ...
├── Lytro-Illum/
│   ├── ...
pred_root/
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


import os
import cv2
import numpy as np
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
import matplotlib.font_manager as fm

font_path = r'C:\Users\123\Desktop\zhiying-desktop\SOD_Map\SimHei.ttf'
fm.fontManager.addfont(font_path)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 设置全局字体大小
plt.rcParams['font.size'] = 18  # 调整数字以适应你的需求
#我们可以通过设置不同的阈值（从0到255）来计算每个阈值下的精确度和召回率，最终形成PR曲线。
def PR(pred_folder,gt_folder):
    # 目标图像大小
    target_size = (256, 256)  # 替换为你的目标大小 (width, height)

    # 获取文件夹中的文件名（去掉扩展名）
    pred_files = {os.path.splitext(f)[0] for f in os.listdir(pred_folder) if f.endswith('.jpg') or f.endswith('.png')}
    gt_files = {os.path.splitext(f)[0] for f in os.listdir(gt_folder) if f.endswith('.jpg') or f.endswith('.png')}

    # 找出两者共有的文件名
    common_files = pred_files & gt_files

    # 初始化列表用于存储所有图像的预测值和真实值
    all_pred = []
    all_gt = []

    # 读取图像并计算预测值和真实值
    for file in common_files:
        pred_path = os.path.join(pred_folder, file + '.png')
        gt_path = os.path.join(gt_folder, file + '.png')

        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if pred_img is None or gt_img is None:
            continue

        # 调整图像大小
        pred_img = cv2.resize(pred_img, target_size)
        gt_img = cv2.resize(gt_img, target_size)

        # 将预测图像和真实图像展平并添加到列表中
        all_pred.extend(pred_img.flatten())
        all_gt.extend(gt_img.flatten())

    # 转换为numpy数组
    all_pred = np.array(all_pred)
    all_gt = np.array(all_gt)

    # 将标签从255转换为1（显著性图）
    all_gt = all_gt // 255

    # 初始化precision和recall列表
    precisions = []
    recalls = []

    # 计算不同阈值下的precision和recall
    thresholds = range(0, 256, 1)
    for threshold in thresholds:
        # 将预测值二值化
        pred_bin = (all_pred >= threshold).astype(int)
        
        # 计算TP, FP, FN
        tp = np.sum((pred_bin == 1) & (all_gt == 1))
        fp = np.sum((pred_bin == 1) & (all_gt == 0))
        fn = np.sum((pred_bin == 0) & (all_gt == 1))

        # 计算precision和recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (fn + tp) if (fn + tp) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
    return recalls,precisions

if __name__=='__main__':
    # 你的真实标签路径
    gt_root = r'C:\Users\123\Desktop\zhiying-desktop\SOD_Map\GT'
    # 所有方法的预测结果文件夹根目录
    pred_root = r'C:\Users\123\Desktop\zhiying-desktop\SOD_Map'
    datasets=['DUTLF-FS','HFUT','Lytro-Illum']
    colors = [ 'g', 'b', 'm', 'c', 'y', 'orange','r']
    for dataset in datasets:
        fontsize=18
        plt.figure(figsize=(6, 6))
        plt.xlabel('召回率',fontsize=fontsize)
        plt.ylabel('精确率',fontsize=fontsize)
        i=0
        for method in ['DLGRG','GFRNet','ISNet','LFTransNet','MoLF','TLFNet','Ours']:
            pred_folder = os.path.join(pred_root,method,dataset) 
            gt_folder=os.path.join(gt_root,dataset) 
            recalls,precisions=PR(pred_folder=pred_folder,gt_folder=gt_folder)
            # 使用插值来平滑曲线
            interp_recall = np.linspace(min(recalls), max(recalls), 5)
            interp_precision = interp1d(recalls, precisions, kind='linear')(interp_recall)
            # 绘制平滑的PR曲线
            color=colors[i]
            plt.plot(interp_recall, interp_precision,  linestyle='-',color=color,label=method)
            i=i+1
            print(f'{method} of {dataset} is ok!')
        # 显示网格，0.3透明度
        plt.grid(True, color='gray', alpha=0.3)
        # 设置上边框和右边框为黑色实线
        ax = plt.gca()
        ax.spines['top'].set_color((0.3, 0.3, 0.3, 0.3))  # 灰色，透明度50%
        ax.spines['right'].set_color((0.3, 0.3, 0.3, 0.3))  # 灰色，透明度50%
        ax.spines['top'].set_linewidth(1)
        ax.spines['right'].set_linewidth(1)
        ax.spines['top'].set_linestyle('-')
        ax.spines['right'].set_linestyle('-')
        # 调整布局
        # plt.tight_layout()
        # 显示图例
        plt.legend(loc='lower left',fontsize=fontsize)
        #保存
        savepath=os.path.join(r'C:\Users\123\Desktop\zhiying-desktop\SOD_Map\多模态论文\PR',f'{dataset}.png')
        plt.savefig(savepath)
        plt.close()
        