'''
功能：按图片顺序将HUNT重命名为0000x.png
'''

import os
import re
# 设置路径
folder = r"E:\zhiying\LF\dataset\HFUT\GT352"  # 替换成你的文件夹路径
# 匹配形如 '12.png', '2.png' 的文件
def extract_number(filename):
    match = re.match(r"(\d+)", os.path.splitext(filename)[0])
    return int(match.group(1)) if match else -1
# 获取所有 png 文件
files = [f for f in os.listdir(folder)]
files.sort(key=extract_number)
file_type=files[0].split('.')[-1]
# 重命名为 00001.png 这种格式
for idx, filename in enumerate(files):
    new_name = f"{idx+1:05d}.{file_type}"  # 00001.png, 00002.png, ...
    old_path = os.path.join(folder, filename)
    new_path = os.path.join(folder, new_name)
    os.rename(old_path, new_path)
print("重命名完成！")
