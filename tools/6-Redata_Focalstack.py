'''
功能：所有的Focalstack图片都在一个文件夹中，先将根据场景分成若干个文件夹，再将每个场景转换为.mat数据
dataset:处理的数据集，可选['DUTLF-FS','HFUT','LytroIllum','PKU-LF']
size：.mat数据的分辨率
请注意：
1.DUTLF-FS：在alljpg2classjpg时要过滤掉['1002','1081','1082']，这个focalstack是多的
2.Lytro Illum：a.全聚焦图像中有一张图片是jpg，最好改成png保持统一。
                b.在classjpg2mat时,开启Lytro Illum的专用排序机制，保证深度从浅到深
3.PKU-LF:在classjpg2mat时,开启PKU-LF的专用排序机制，保证深度从浅到深
'''

import scipy.io as sio
import os
import PIL.Image
import numpy as np
import shutil
from tqdm import tqdm
import re
#将focal图像分成1100个文件夹，每个文件夹里面只有自己的focal图片
def alljpg2classjpg(data_root,save_dir,focals_list,dataset):

    i=0
    for focal in tqdm(focals_list):   # focal='0001__refocus_00.jpg'
        name=focal.split('_')[0]
        # DUTLF-FS专用
        if dataset=='DUTLF-FS' and name in ['1002','1081','1082']:
            continue
        i=i+1
        if not os.path.exists(os.path.join(save_dir,name)):#-16
            os.mkdir(os.path.join(save_dir,name))#-16
        image_path=os.path.join(data_root,focal)
        img = PIL.Image.open(image_path).convert('RGB')
        save_path = os.path.join(save_dir,name,focal)#-16
        img.save(save_path)
    print(str(i)+' all_jpg to class_jpg ok! ')

def PKULF_value(filename):
    match = re.search(r'f([+-][\d\.]+)', filename)
    match=match.group(1)[:-1]
    return float(match) if match else 0.0

def DUTLF_value(filename):
    # 匹配形如 refocus_2 或 refocus_10 的部分
    match = re.search(r'refocus_(\d+)', filename)
    return int(match.group(1)) if match else 0


def classjpg2mat(data_root,save_dir,focals_list,size,dataset):
    j=0
    for focal in tqdm(focals_list):   #focal='0001'
        j=j+1
        focals_dir = os.path.join(data_root,focal)  #focal_dir='train_focals_classjpg/0001'
        images_list = sorted(os.listdir(focals_dir))
        if len(images_list)>10:
            debug=1
        # PKU-LF排序专用
        if dataset=='DUTLF-FS' or 'HFUT' or 'LytroIllum':
            images_list = sorted(images_list, key=DUTLF_value)  
        elif dataset=='PKU-LF':
            images_list = sorted(images_list, key=PKULF_value)  
        focal_mat = None
        for i in range(12):
            # k=i%len(images_list)      #按顺序来
            # k=int((i+1)/12*len(images_list)-1)  # mat按深度从浅到深排序图片
            k=min(i,len(images_list)-1)
            image = images_list[k] 
            image_path = os.path.join(focals_dir,image) #image_dir='train_focals_classjpg/0001/0001__refocus_00.jpg'
            img = PIL.Image.open(image_path).convert('RGB')
            img = img.resize((size, size))
            img = np.array(img, dtype=np.uint8)
            if focal_mat is None:
                focal_mat = img
            else:
                focal_mat = np.dstack([focal_mat,img])
        save_path = os.path.join(save_dir,focal.replace('__refocus_', '') + '.mat')
        sio.savemat(save_path, {'img':focal_mat})
    print(str(j)+' class_jpg to mat ok! ')

if __name__ == '__main__': 
    
    dataset='DUTLF-FS'  #['DUTLF-FS','HFUT','LytroIllum','PKU-LF']
    size=352    #[256,352]
    data_root = r'E:\zhiying\LF\dataset\DUTLF-FS\TrainingSet\focalstack'    #focalstack_path
    #将all_jpg转成class_jpg 
    save_dir = data_root+'_class'
    focals_list = os.listdir(data_root) 
    # if os.path.exists(save_dir): 
    #     shutil.rmtree(save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    alljpg2classjpg(data_root,save_dir,focals_list,dataset)  



    #将class_jpg转成mat  
    #代码需要修改名字
    class_root = data_root+'_class'
    save_dir = data_root+f'_mat_{size}'
    focals_list = os.listdir(class_root)
    # if os.path.exists(save_dir): 
    #     shutil.rmtree(save_dir)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    classjpg2mat(class_root,save_dir,focals_list,size=size,dataset=dataset)
    

    # k_dir=r'E:\zhiying\LF\dataset\test\LytroIllum\test_masks256'
    # k_list = os.listdir(k_dir)  # 0018.png
    # for focal in focals_list:   # 0018
    #     focal_name=focal+'.png'
    #     if focal_name not in k_list:
    #         print('{}\tfocal_name not in gts_list'.format(focal))
    # for k in k_list:
    #     k_name=k.replace('.png','')
    #     if k_name not in focals_list:
    #         print('{}\tgt_name not in focals_list'.format(k_name))
    # print('ok')
