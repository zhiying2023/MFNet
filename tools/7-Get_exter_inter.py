'''
功能：利用腐蚀和膨胀获取GT的内轮廓和外轮廓
'''

import os
import cv2
import numpy as np
# 设置路径
data_path = '/workspace/dataset/train/train_masks'
dst_path1 = '/workspace/dataset/train/train_edge_external/'
dst_path2 = '/workspace/dataset/train/train_edge_internal/'

# 创建输出目录（如果不存在）
os.makedirs(dst_path1, exist_ok=True)
os.makedirs(dst_path2, exist_ok=True)

# 获取图像文件列表
img_files = [f for f in os.listdir(data_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

# 创建结构元素，相当于 MATLAB 中的 strel('disk', 5)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))  # 直径为 11 的椭圆形结构元素

for name in img_files:
    # 读取掩码图像
    mask_path = os.path.join(data_path, name)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 确保掩码为二值图像
    _, mask_bin = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 膨胀和腐蚀操作
    dilated = cv2.dilate(mask_bin, kernel, iterations=1)
    eroded = cv2.erode(mask_bin, kernel, iterations=1)

    # 计算外部和内部边缘
    edge_external = 255 - cv2.subtract(dilated, mask_bin)
    edge_internal = cv2.subtract(mask_bin, eroded)

    # 保存边缘图像
    cv2.imwrite(os.path.join(dst_path1, name), edge_external)
    cv2.imwrite(os.path.join(dst_path2, name), edge_internal)

