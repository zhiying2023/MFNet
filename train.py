import logging
import os
from datetime import datetime
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
# from torch.utils.tensorboard import SummaryWriter
from utils.data_depth_edge import get_loader, test_dataset
from MFNet import LF as model
from options import opt
from utils.utils import clip_gradient, adjust_lr
from utils.pytorch_utils import Save_Handle
from torch.autograd import Variable
import sys
import matplotlib.pyplot as plt

code_name=os.path.basename(sys.argv[0]).split('.')[0]

torch.cuda.set_device(2)
print("GPU available:", torch.cuda.is_available())
cudnn.benchmark = True
torch.cuda.current_device()
save_list = Save_Handle(max_num=1)


def print_network(model, name):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()#numel用于返回数组中的元素个数
    print(name)
    print('The number of parameters:{}'.format(num_params))
    return num_params
start_epoch = 0

model = model()
if (opt.load_mit is not None):
    model.focal_encoder._load_state_dict(opt.load_mit)
    model.rgb_encoder._load_state_dict(opt.load_mit)
    model.depth_encoder._load_state_dict(opt.load_mit)
else:
    print("No pre-trian!")

model.cuda()
params = model.parameters()
optimizer = torch.optim.AdamW(params, opt.lr, weight_decay=1e-4)  #weight_decay正则化系数
model_params = print_network(model, 'lf_pvt')
rgb_root = opt.rgb_root
gt_root = opt.gt_root
fs_root = opt.fs_root
depth_root = opt.depth_root
edgein_root = opt.edgein_root
edgeex_root = opt.edgeex_root
test_rgb_root = opt.test_rgb_root
test_fs_root = opt.test_fs_root
test_gt_root = opt.test_gt_root
test_depth_root = opt.test_depth_root
save_path = opt.save_path

if not os.path.exists(os.path.join(save_path,'log')):
    os.makedirs(os.path.join(save_path,'log'))
if not os.path.exists(os.path.join(save_path,code_name)):
    os.makedirs(os.path.join(save_path,code_name))

#load data
print('load data...')
train_loader = get_loader(rgb_root, gt_root, fs_root, depth_root, edgein_root, edgeex_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_rgb_root, test_gt_root, test_fs_root, test_depth_root, testsize=opt.trainsize)
total_step = len(train_loader)
logging.basicConfig(filename=os.path.join(save_path,'log',code_name+'.log'), format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate,  save_path,
        opt.decay_epoch))


#set loss function
# CE = torch.nn.BCEWithLogitsLoss()


def structure_loss(pred, mask):
    weit = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3))/weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

# loss_edge
def FocalLoss_in(pred, mask, weight=None, gamma=2.0, alpha=0.25, reduction='mean', avg_factor=None):
    pred_sigmod = torch.sigmoid(pred)
    pt = (1-pred_sigmod) * mask + pred_sigmod * (1 - mask)
    focal_weight = (alpha * mask + (1 - alpha) * (1 - mask)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, mask, reduction='none') * focal_weight
    return loss.mean()

def FocalLoss_ex(pred, mask, weight=None, gamma=2.0, alpha=0.25, reduction='mean', avg_factor=None):
    pred_sigmod = torch.sigmoid(pred)
    pred = 1-pred
    mask = 1-mask
    pt = (1-pred_sigmod) * mask + pred_sigmod * (1 - mask)
    focal_weight = (alpha * mask + (1 - alpha) * (1 - mask)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, mask, reduction='none') * focal_weight
    return loss.mean()


step = 0

best_mae = 1
best_epoch = 0

# resume=opt.resume
# if resume==True:
#     checkpoint= torch.load('xxx.pth')
#     model.load_state_dict(checkpoint['model_state_dict'],strict=True)
#     start_epoch=checkpoint['epoch']
#     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def train(train_loader, model, optimizer, epoch, save_path,train_loss_list):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    for i, (images, gts, focal, depths, edge_internal, edge_external) in enumerate(train_loader, start=1):
        basize, dim, height, width = focal.size()
        gts = gts.cuda()
        depths = depths.cuda()
        edge_internal=edge_internal.cuda()
        edge_external=edge_external.cuda()
        images, gts, focal, depths = Variable(images), Variable(gts), Variable(focal), Variable(depths)
        focal = focal.view(1, basize, dim, height, width).transpose(0, 1)  # (basize, 1, 36, 256, 256)
        focal = torch.cat(torch.chunk(focal, chunks=12, dim=2), dim=1)  # (basize, 12, 3, 256, 256)
        focal = torch.cat(torch.chunk(focal, chunks=basize, dim=0), dim=1)  # (1, basize*12, 6, 256, 256)
        focal = focal.view(-1, *focal.shape[2:])  # [basize*12, 6, 256, 256)
        focal = focal.cuda()
        images = images.cuda()
        optimizer.zero_grad()
        output,pred1,pred2,pred3,pred_in,pred_ex = model(focal, images, depths)
        
        loss1 = structure_loss(output, gts)+structure_loss(pred1, gts)+structure_loss(pred2, gts)+structure_loss(pred3, gts)
        loss2 = FocalLoss_in(pred_in, edge_internal)+FocalLoss_ex(pred_ex, edge_external)
        loss = loss1 +loss2
        loss.backward()
        # 梯度裁剪
        clip_gradient(optimizer, opt.clip)
        optimizer.step()
        step += 1
        epoch_step += 1
        loss_all += loss.data
    loss_all /= epoch_step
    # 训练中断保留参数
    train_loss_list.append(loss_all.detach().cpu())
    return train_loss_list



def test(test_loader, model, epoch, save_path,test_acc_list):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, focal, gt, depth, name = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            dim, height, width = focal.size()
            basize = 1

            focal = focal.view(1, basize, dim, height, width).transpose(0, 1)  # (basize, 1, 36, 256, 256)
            focal = torch.cat(torch.chunk(focal, chunks=12, dim=2), dim=1)  # (basize, 12, 3, 256, 256)
            focal = torch.cat(torch.chunk(focal, chunks=basize, dim=0), dim=1)  # (1, basize*12, 6, 256, 256)
            focal = focal.view(-1, *focal.shape[2:])
            focal = focal.cuda()
            image = image.cuda()
            depth = depth.cuda().unsqueeze(0)

            res,pred1,pred2,pred3,pred_in,pred_ex = model(focal, image, depth)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save({
                        'epoch': epoch,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'model_state_dict': model.state_dict()
                        }, 
                        os.path.join(save_path,code_name,"{}_best.pth".format(code_name)))
        torch.save({
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'model_state_dict': model.state_dict()
                    }, 
                    os.path.join(save_path,code_name,"{}_best.pth".format(code_name)))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        test_acc_list.append(mae)
        return test_acc_list


if __name__ == '__main__':
    logging.info("Start train...")
    # 初次衰减循环增大10个epoch即110后才进行第一次衰减
    train_loss_list=[]
    test_acc_list=[]
    for epoch in range(start_epoch, opt.epoch+1):
        # if (epoch % 50 ==0 and epoch < 60):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        # writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train_loss_list=train(train_loader, model, optimizer, epoch, save_path,train_loss_list)
        test_acc_list=test(test_loader, model, epoch, save_path,test_acc_list)
    logging.info('#train loss#:{}'.format(train_loss_list))
    logging.info('#test acc#:{}'.format(test_acc_list))
    #可视化
    epochs = list(range(1, len(train_loss_list) + 1))
    # 保存训练损失曲线
    plt.figure()
    plt.plot(epochs, train_loss_list, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(os.path.join(save_path,code_name,'training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()  # 关闭当前图像，避免影响后续绘图
    # 保存测试准确率曲线
    plt.figure()
    plt.plot(epochs, test_acc_list, 'r-', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Test Accuracy')
    plt.legend()
    plt.savefig(os.path.join(save_path,code_name,'test_accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()  # 关闭当前图像
