'''
功能：深度图片BMP改成PNG
'''

from PIL import Image
import os

import os
from PIL import Image

def batch_convert_bmp_to_png(folder_path):
    for fname in os.listdir(folder_path):
        if fname.lower().endswith(".bmp"):
            bmp_path = os.path.join(folder_path, fname)
            png_path = os.path.join(folder_path, fname.replace(".bmp", ".png"))
            img = Image.open(bmp_path)
            img.save(png_path)
            print(f"已转换: {fname} -> {os.path.basename(png_path)}")

batch_convert_bmp_to_png(r'E:\zhiying\LF\dataset\test\HFUT\test_depths352')