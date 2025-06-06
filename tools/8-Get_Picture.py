'''
功能：再Eval.py筛选出可视化图后，根据图片名字，找出所有方法的对应的图片
推荐路径：
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

import os
from PIL import Image
import shutil

# 总文件夹路径（里面包含多个方法子文件夹）
root_folder = r"C:\Users\123\Desktop\zhiying-desktop\SOD_Map"
# 指定要查找的图片名（不带扩展名）
dataset='Lytro-Illum'
# target_names = ["4024", "4140", "4150",'4178','4274','4309','4314','4563','4574','5041','5222','5238','5392','4306']  # 可以替换为你要找的图片名列表
# target_names = ["4234",'4306','4409','4854','5211'] #test-abcde
target_names = ["3843",'3871','3883','4024','4339','4371','4426','4473','4532','4986','5131','5314'] #test-de
# dataset='DUTLF-FS'
# target_names = ['0176', "0203", '0668','1257','1259']
# dataset='HFUT'
# target_names = ['00008', "00032"]
# 新文件夹保存路径
output_folder = r"E:\zhiying\LF\results-selected"
os.makedirs(output_folder, exist_ok=True)

# 遍历每个方法子文件夹
for method in os.listdir(root_folder):
    # if method not in ['GT','AiF','Ours','test1a-heatmap','test1b-heatmap','test1c-heatmap','test1d-heatmap','test1e-heatmap'] :   
    # if method in ['test1a-heatmap','test1b-heatmap','test1c-heatmap','test1d-heatmap','test1e-heatmap','edgeinmaps','edgeexmaps','test1d-testmap','test1e-testmap'] :
    # if method not in ['GT','AiF','edgeinmaps','edgeexmaps','test1d-testmap','test1e-testmap'] : 
    #     continue
    method_path = os.path.join(root_folder, method)
    if not os.path.isdir(method_path):
        continue

    for name in target_names:
        found = False
        for ext in ['.jpg', '.png']:
            if dataset=='Lytro-Illum':
                filename = 'IMG_'+name + ext
            else:
                filename = ''+name + ext
            file_path = os.path.join(method_path,dataset, filename)
            if os.path.exists(file_path):
                # 打开图片并保存为 jpg 格式（统一格式）
                img = Image.open(file_path).convert("RGB")
                save_name = f"{name}_{method}.jpg"
                save_path = os.path.join(output_folder, save_name)
                img.save(save_path)
                print(f"保存：{save_name}")
                found = True
                break  # 如果找到了就不再试另一个扩展名
        if not found:
            print(f"未找到：{name} in {method}")
