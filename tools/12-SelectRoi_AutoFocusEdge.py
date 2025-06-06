'''
功能：自动聚焦算法的实现
saliency_map_root：显著图路径
all_in_focus_image_root：全聚焦图像路径
focalstack_root：.mat格式焦点堆栈图像路径
roi_size：感兴趣区域大小
target_size：图像统一上采样分辨率
'''


import cv2
import numpy as np
import os
from scipy import io as sio
from einops import rearrange
from PIL import Image
from tqdm import tqdm
'----------------------------select roi---------------------------------------'
def get_largest_connected_region(binary_mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if num_labels <= 1:
        return binary_mask
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    largest_region_mask = (labels == largest_label).astype(np.uint8)
    return largest_region_mask
def get_best_roi_from_saliency(saliency_map_path, all_in_focus_image_path, roi_size=224, target_size=1080):
    saliency_map = cv2.imread(saliency_map_path)
    all_in_focus_image = cv2.imread(all_in_focus_image_path)
    # 上采样显著图和全聚焦图像到指定分辨率
    saliency_map = cv2.resize(saliency_map, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    all_in_focus_image = cv2.resize(all_in_focus_image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    # 灰度化显著图用于计算阈值
    saliency_gray = cv2.cvtColor(saliency_map, cv2.COLOR_BGR2GRAY) if saliency_map.ndim == 3 else saliency_map
    threshold = 2 * np.mean(saliency_gray)
    binary_mask = (saliency_gray >= threshold).astype(np.uint8) 
    # 提取最大连通区域
    largest_region_mask = get_largest_connected_region(binary_mask)   
    # 计算 Canny 边缘响应（基于灰度的全聚焦图像）
    all_in_focus_gray = cv2.cvtColor(all_in_focus_image, cv2.COLOR_BGR2GRAY) if all_in_focus_image.ndim == 3 else all_in_focus_image
    masked_focus = cv2.bitwise_and(all_in_focus_gray, all_in_focus_gray, mask=largest_region_mask)
    edges = cv2.Canny(masked_focus, 50, 150)
    # 滑动窗口寻找平均梯度最大的 ROI
    step = 10
    max_grad = -1
    best_coord = (0, 0)
    for i in range(0, target_size - roi_size + 1, step):
        for j in range(0, target_size - roi_size + 1, step):
            roi_mask = largest_region_mask[i:i+roi_size, j:j+roi_size]
            noroi_mask=(1-largest_region_mask)[i:i+roi_size, j:j+roi_size]
            # if np.sum(roi_mask) > np.sum(largest_region_mask)* 0.1 or np.sum(roi_mask)>=np.sum(noroi_mask):#选中区域的有效性，大roi，小roi
            roi_edge = edges[i:i+roi_size, j:j+roi_size]
            mean_grad = np.mean(roi_edge)
            if mean_grad > max_grad:
                max_grad = mean_grad
                best_coord = (i, j)
    x, y = best_coord
    best_roi = all_in_focus_image[x:x+roi_size, y:y+roi_size]

    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # 画红框：颜色为红色(BGR = 0, 0, 255)，线条粗细为2
    cv2.rectangle(edges_color, (y,x), (y+roi_size, x+roi_size), (0, 0, 255), 5)
    cv2.imwrite(f"./result/Edge/{name}.png", edges_color)
    # cv2.imwrite(f"./result/AutoFocus/{name}_roi.png", best_roi) 

    return best_roi, (x, y), largest_region_mask



'----------------------------autofouces---------------------------------------'
# 加载和预处理图像
def load_image_as_tensor(img_path,coord,roi_size):
    with open(img_path, 'rb') as f:
        focal = sio.loadmat(f)
        focal = focal['img']
    # 之前保存时候忘记转为rgb了
    # focal = rearrange(focal, 'h w (n c) -> n h w c', c=3)  # 变成 (N, H, W, C)   
    # focal = np.stack([np.array(Image.fromarray(img.astype(np.uint8)).convert('RGB')) for img in focal])
    # focal = rearrange(focal, 'n h w c -> h w (n c)')
    # print(focal.shape)
    # img = image.img_to_array(focal)   

    return focal
# 单张图片预测
def predict(img_path,coord,roi_size,target_size):
    img_original = load_image_as_tensor(img_path,coord,roi_size)
    imgs = cv2.resize(img_original, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    imgs=rearrange(imgs,'h w (n c)->n h w c',c=3)
    img_original=rearrange(img_original,'h w (n c)->n h w c',c=3)
    imgs=imgs[:,coord[0]:coord[0]+roi_size, coord[1]:coord[1]+roi_size,:]
    all_in_focus_gray=np.zeros((imgs.shape[0],imgs.shape[1],imgs.shape[2]))
    for i in range(imgs.shape[0]):
        all_in_focus_gray[i] = cv2.Canny(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY), 50, 150)
    means=all_in_focus_gray.mean(axis=(1, 2))
    max_index = np.argmax(means)

    mean_str = np.array2string(means, separator=', ', max_line_width=np.inf)
    # print(img_path)
    # print(mean_str)
    # print(max_index)
    with open('./log.txt', 'a+') as f:    
        f.write(f"img_path:{img_path}\nbest_coord:{coord}\nmean_edge: {mean_str}\nmax_index: {max_index}\n")

    return img_original[max_index]

if __name__=='__main__':
    if os.path.exists('./log.txt'):
        os.remove('./log.txt')
    saliency_map_root=r'C:\Users\123\Desktop\zhiying-desktop\SOD_Map\Ours\PKU-LF'
    all_in_focus_image_root=r'E:\zhiying\LF\dataset\test\PKU-LF\test_images'
    focalstack_root=r'E:\zhiying\LF\dataset\test\PKU-LF\test_mat_original'
    roi_size=224
    target_size=1080
    if not os.path.exists('./result/AutoFocus'):
        os.makedirs('./result/AutoFocus')
    if not os.path.exists('./result/Edge'):
        os.makedirs('./result/Edge')

    saliency_map_list = [f for f in os.listdir(saliency_map_root) if f.endswith('.jpg') or f.endswith('.png')]
    all_in_focus_image_list = [f for f in os.listdir(all_in_focus_image_root) if f.endswith('.jpg') or f.endswith('.png')]
    focalstack_list = [f for f in os.listdir(focalstack_root) if f.endswith('.mat')]
    saliency_map_list = sorted(saliency_map_list)
    all_in_focus_image_list = sorted(all_in_focus_image_list)
    focalstack_list = sorted(focalstack_list)

    for i in tqdm(range(len(focalstack_list))):
        name=saliency_map_list[i].split('.')[0]
        saliency_map_path=os.path.join(saliency_map_root,saliency_map_list[i])
        all_in_focus_image_path=os.path.join(all_in_focus_image_root,all_in_focus_image_list[i])
        focalstack_path=os.path.join(focalstack_root,focalstack_list[i])
        # print(saliency_map_path,all_in_focus_image_path,focalstack_path)
        '----------------------------select roi---------------------------------------'
        roi, coord, mask = get_best_roi_from_saliency(saliency_map_path=saliency_map_path,
                                                    all_in_focus_image_path=all_in_focus_image_path,
                                                    roi_size=roi_size,
                                                    target_size=target_size)
        # print("最佳ROI位置（左上角坐标）:", coord)
         

        '----------------------------autofouces---------------------------------------'
        autofoucus_img=predict(focalstack_path,coord,roi_size,target_size)
        cv2.imwrite(f"./result/AutoFocus/{name}.png", cv2.cvtColor(autofoucus_img, cv2.COLOR_RGB2BGR)) 