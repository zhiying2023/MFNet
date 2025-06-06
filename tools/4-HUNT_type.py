'''
功能：检测对图片格式为L的GT或者Depth转换后，保持格式不变
'''
from PIL import Image
import numpy as np

def check_depth_image(path):
    with Image.open(path) as img:
        print("PIL 图像模式:", img.mode)
        arr = np.array(img)
        print("Numpy dtype:", arr.dtype)
        print("最小值:", arr.min(), "最大值:", arr.max())

check_depth_image(r'E:\zhiying\LF\dataset\train\train_depths352\00002.png')