'''
功能：从完整的HUNT中划分出训练集和测试集
folder1：测试集文件，可以找别人论文的HUNT显著性预测图即可
folder2：完整的HUNT文件
matched_output：HUNT测试文件
unmatched_output：HUNT训练文件

'''
import os
import shutil
from tqdm import tqdm
def split_matched_images(folder1, folder2, matched_dir, unmatched_dir):
    # 创建输出文件夹
    os.makedirs(matched_dir, exist_ok=True)
    os.makedirs(unmatched_dir, exist_ok=True)

    # 获取 folder1 的所有文件名集合（不含路径）
    list1=os.listdir(folder1)
    folder1_files = set(list1)

    
    # 遍历 folder2 中的文件
    for file_name in tqdm(os.listdir(folder2)):
        file1_type=list1[0].split('.')[-1]
        file2_type=file_name.split('.')[-1]
        src_path = os.path.join(folder2, file_name)
        if not os.path.isfile(src_path):
            continue  # 跳过子目录等

        if file_name.replace(file2_type,file1_type) in folder1_files:
            # 有相同文件名 → 放入 matched 文件夹
            dst_path = os.path.join(matched_dir, file_name)
        else:
            # 没有相同文件名 → 放入 unmatched 文件夹
            dst_path = os.path.join(unmatched_dir, file_name)

        shutil.copy2(src_path, dst_path)  # 如果你想移动而不是复制，用 shutil.move

        # print(f"{file_name} -> {'matched' if file_name in folder1_files else 'unmatched'}")

# 示例路径（替换为你的实际路径）
folder1 = r"E:\zhiying\LF\dataset\test\HFUT\test_depths"    #测试集部分文件
folder2 = r"E:\zhiying\LF\dataset\HFUT\Depth352"  #完整的文件
matched_output = r"E:\zhiying\LF\dataset\HFUT\Depth352-test"
unmatched_output = r"E:\zhiying\LF\dataset\HFUT\Depth352-train"

split_matched_images(folder1, folder2, matched_output, unmatched_output)
