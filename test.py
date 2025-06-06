import torch
import torch.nn.functional as F
import sys

import numpy as np
import os, argparse
import cv2
from utils.data_depth_edge import test_dataset
from MFNet import LF as model
from torchvision.utils import save_image
from Metric.saliency.eval_evaluator import Eval_thread
from Metric.saliency.eval_dataloader import EvalDataset
from tqdm import tqdm
print("GPU available:", torch.cuda.is_available())


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='./dataset/test',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path


#load the model
model = model()
model.cuda()

# DUTLF,HUNT,Lytro Illum
checkpoint= torch.load('xxx.pth')
test_datasets = ['DUTLF-FS','HFUT','Lytro-Illum']
model.load_state_dict(checkpoint['model_state_dict'],strict=True)
model.eval()

#热力图
def CAM(features, img_path, save_path):
    # features.retain_grad()
    # grads = features.grad
    features = features.squeeze(0)
    heatmap = features.detach().cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = np.uint8(heatmap * 0.5 + img * 0.5)
    cv2.imwrite(save_path, superimposed_img)



import shutil
# if os.path.exists(os.path.join('/workspace/test/heatmaps')):
    #     shutil.rmtree('/workspace/test/heatmaps', ignore_errors=True)
for dataset in test_datasets:
    save_path = './test_best/testmaps/' + dataset 
    save_path_heatmap = './test_best/heatmaps/' + dataset 
    save_path_edgein = './test_best/edgeinmaps/' + dataset 
    save_path_edgeex = './test_best/edgeexmaps/' + dataset 
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not os.path.exists(save_path_heatmap):
        os.makedirs(save_path_heatmap)

    if not os.path.exists(save_path_edgein):
        os.makedirs(save_path_edgein)

    if not os.path.exists(save_path_edgeex):
        os.makedirs(save_path_edgeex)

    test_rgb_root = os.path.join(dataset_path , dataset , 'test_images/')
    test_gt_root = os.path.join(dataset_path , dataset , 'test_masks/')
    test_fs_root = os.path.join(dataset_path , dataset , 'test_focals/')
    test_depth_root= os.path.join(dataset_path , dataset , 'test_depths/')
    test_loader = test_dataset(test_rgb_root, test_gt_root, test_fs_root, test_depth_root, testsize=opt.testsize)
    for i in tqdm(range(test_loader.size)):
        #todo 位置
        # image, focal, gt, name = test_loader.load_data()
        image, focal, gt, depth, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        dim, height, width = focal.size()
        basize = 1
        focal = focal.view(1, basize, dim, height, width).transpose(0, 1)  # (basize, 1, 36, 256, 256)
        focal = torch.cat(torch.chunk(focal, chunks=12, dim=2), dim=1)  # (basize, 12, 3, 256, 256)
        focal = torch.cat(torch.chunk(focal, chunks=basize, dim=0), dim=1)  # (1, basize*12, 6, 256, 256)
        focal = focal.view(-1, *focal.shape[2:])  # [basize*12, 6, 256, 256)
        focal = focal.cuda()
        image = image.cuda()
        depth = depth.cuda().unsqueeze(0)
        res,pred1,pred2,pred3,pred_in,pred_ex = model(focal, image, depth)

        sod=res.detach()
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min())
        cv2.imwrite(os.path.join(save_path , name), res * 255)

        img_path = os.path.join(test_rgb_root,name)
        if dataset=='Lytro-Illum':
            img_path = img_path.split('.')[0]+'.png'
        else:
            img_path = img_path.split('.')[0]+'.jpg'
        CAM(sod, img_path, os.path.join(save_path_heatmap , name))

        pred_in = pred_in.sigmoid().data.cpu().numpy().squeeze()
        pred_in = (pred_in - pred_in.min()) / (pred_in.max() - pred_in.min() + 1e-8)
        pred_in=1-pred_in
        cv2.imwrite(os.path.join(save_path_edgein , name), pred_in * 255)

        pred_ex = pred_ex.sigmoid().data.cpu().numpy().squeeze()
        pred_ex = (pred_ex - pred_ex.min()) / (pred_ex.max() - pred_ex.min() + 1e-8)
        pred_ex=1-pred_ex
        cv2.imwrite(os.path.join(save_path_edgeex , name), pred_ex * 255)

    loader = EvalDataset(save_path, test_gt_root)
    #1.将可视化保存到savepath
    #2.将每个数据集的每一张图片mae保存到路径(xlsxpath)，excle名字(method.xlsx)，子表名为各数据集(dataset)
    thread = Eval_thread(loader, 
                         savepath='/workspace/lfsod_savepath/log/log.txt',
                         dataset=dataset, 
                         cuda=False,
                         method='ours')
    # thread = Eval_thread(loader, 
    #                      savepath='/workspace/lfsod_savepath/log/log.txt',
    #                      xlsxpath='/workspace/test_best',
    #                      dataset=dataset, 
    #                      cuda=False,
    #                      method='ours')
    print(print(thread.run()))
    print('{} Test Done!'.format(dataset))
    
