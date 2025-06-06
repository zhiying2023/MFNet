'''
功能：在不改变图片的格式的前提下，修改图片的分辨率，比如GT他不是RGB图像，格式为L
input_folder：要改变分辨率的路径
'''
import os
from PIL import Image
from tqdm import tqdm
# 原始深度图文件夹路径
input_folder = r"E:\zhiying\LF\dataset\Lytro-Illum\GT(all-in-focus)"  # 替换为你的原图路径
# 输出文件夹路径
size=352
output_folder = input_folder+f"{size}"  # 替换为你想保存的位置

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 遍历所有 PNG 文件
for filename in tqdm(os.listdir(input_folder)):
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".png") or filename.lower().endswith(".bmp"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # 打开图片（按原格式读取）
        img = Image.open(input_path)

        # Resize 成 256x256，使用最近邻插值（常用于深度图）
        img_resized = img.resize((size, size), resample=Image.NEAREST)

        # 保存
        img_resized.save(output_path)

print(f"✅ 所有图已统一为{size}×{size}，并保存到新文件夹！")
