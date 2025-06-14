import os
import time
from datetime import datetime
import numpy as np
import torch
from torchvision import transforms
import cv2
import pandas as pd
class Eval_thread():
    def __init__(self, loader, savepath=False,xlsxpath=False, method='lf',dataset='', cuda=False):
        self.loader = loader
        self.method = method
        self.dataset = dataset
        self.cuda = cuda
        self.result_record={}
        self.savepath=savepath
        self.xlsxpath=xlsxpath
    def run(self, return_value=False):
        mae,max_f,max_e,s=0,0,0,0
        start_time = time.time()
        if self.xlsxpath:
            mae = self.run_mae()
        elif self.xlsxpath==False:
            mae = self.Eval_mae()
            max_f = self.Eval_fmeasure()
            max_e = self.Eval_Emeasure()
            s = self.Eval_Smeasure()
        result_str='[cost:{:.4f}s]{} dataset with {} method get {:.4f} mae, {:.4f} max-fmeasure, {:.4f} max-Emeasure, {:.4f} S-measure..'.format(
            time.time() - start_time, self.dataset, self.method, mae, max_f, max_e, s)
        
        if self.savepath:
            open(self.savepath, 'a+').write(result_str + '\n')
        return result_str
    def run_mae(self, return_value=False):
        mae = self.Eval_mae()
        # 修改，导出结果
        self.result_record={k: v for k, v in sorted(self.result_record.items(), key=lambda item: item[1])}
        df = pd.DataFrame(list(self.result_record.items()), columns=['Image Name', 'Comparison Value'])
        save_path=os.path.join(self.xlsxpath,self.method+'.xlsx')
        if os.path.exists(save_path):
            with pd.ExcelWriter(save_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                df.to_excel(writer, sheet_name=self.dataset, index=False, header=not os.path.exists(save_path), startrow=0)
        else:
            df.to_excel(save_path, sheet_name=self.dataset, index=False, engine='openpyxl')
        return mae



    def Eval_mae(self):
        print('eval[MAE]:{} dataset with {} method.'.format(self.dataset, self.method))
        avg_mae, img_num = 0.0, 0.0
        i=0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    gt = trans(gt)
                mea = torch.abs(pred - gt).mean()
                if mea == mea:  # for Nan
                    avg_mae += mea
                    img_num += 1.0
                # 修改，记录
                name=self.loader.image_path[i].split('/')[-1]
                self.result_record[name]=mea.item()
                i=i+1
            avg_mae /= img_num  
            return avg_mae.item()

    def Eval_fmeasure(self):
        print('eval[FMeasure]:{} dataset with {} method.'.format(self.dataset, self.method))
        beta2 = 0.3
        avg_p, avg_r, img_num = 0.0, 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    gt = trans(gt)
                prec, recall = self._eval_pr(pred, gt, 255)
                avg_p += prec
                avg_r += recall
                img_num += 1.0
            avg_p /= img_num
            avg_r /= img_num
            score = (1 + beta2) * avg_p * avg_r / (beta2 * avg_p + avg_r)
            score[score != score] = 0  # for Nan

            return score.max().item()

    def Eval_Emeasure(self):
        print('eval[EMeasure]:{} dataset with {} method.'.format(self.dataset, self.method))
        avg_e, img_num = 0.0, 0.0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    gt = trans(gt)
                max_e = self._eval_e(pred, gt, 255)
                if max_e == max_e:
                    avg_e += max_e
                    img_num += 1.0

            avg_e /= img_num
            return avg_e

    def Eval_Smeasure(self):
        print('eval[SMeasure]:{} dataset with {} method.'.format(self.dataset, self.method))
        alpha, avg_q, img_num, tmp = 0.5, 0.0, 0.0, 0
        with torch.no_grad():
            trans = transforms.Compose([transforms.ToTensor()])
            for pred, gt in self.loader:
                if self.cuda:
                    pred = trans(pred).cuda()
                    gt = trans(gt).cuda()
                else:
                    pred = trans(pred)
                    gt = trans(gt)
                y = gt.mean()
                if y == 0:
                    x = pred.mean()
                    Q = 1.0 - x
                elif y == 1:
                    x = pred.mean()
                    Q = x
                else:
                    Q = alpha * self._S_object(pred, gt) + (1 - alpha) * self._S_region(pred, gt)
                    if Q.item() < 0:
                        Q = torch.FloatTensor([0.0])
                
                if np.isnan(Q):
                    tmp += 1
                    img = gt.data.cpu().numpy().squeeze()*255.0
                    cv2.imwrite('./test/{}.jpg'.format(tmp), img)
                if not np.isnan(Q):
                    img_num += 1.0
                    avg_q += Q.item()
            avg_q /= img_num
            return avg_q

    def Eval_Smeasure_loss(self):
        # print('eval[SMeasure]:{} dataset with {} method.'.format(self.dataset, self.method))
        alpha, avg_q, img_num, tmp = 0.5, 0.0, 0.0, 0
        
        # trans = transforms.Compose([transforms.ToTensor()])
        for pred, gt in self.loader:
            if self.cuda:
                pred = (pred).cuda()
                gt = (gt).cuda()
            else:
                pred = (pred)
                gt = (gt)
            y = gt.mean()
            if y == 0:
                x = pred.mean()
                Q = 1.0 - x
            elif y == 1:
                x = pred.mean()
                Q = x
            else:
                Q = alpha * self._S_object(pred, gt) + (1 - alpha) * self._S_region(pred, gt)
                if Q.item() < 0:
                    # Q = torch.FloatTensor([0.0])
                    Q = torch.max(torch.tensor(0).cuda(), Q)

            # if np.isnan(Q):
            #     tmp += 1
            #     img = gt.data.cpu().numpy().squeeze() * 255.0
            #     cv2.imwrite('./test/{}.jpg'.format(tmp), img)

            img_num += 1.0
            try:
                avg_q += Q
            except:
                print(Q)
                exit()
        avg_q /= img_num
        return avg_q

    def LOG(self, output):
        with open(self.logfile, 'a') as f:
            f.write(output)

    def _eval_e(self, y_pred, y, num):
        if self.cuda:
            score = torch.zeros(num).cuda()
        else:
            score = torch.zeros(num)
        for i in range(num):
            fm = y_pred - y_pred.mean()
            gt = y - y.mean()
            align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
            enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
            score[i] = torch.sum(enhanced) / (y.numel() - 1 + 1e-20)
        return score.max()

    def _eval_pr(self, y_pred, y, num):
        if self.cuda:
            prec, recall = torch.zeros(num).cuda(), torch.zeros(num).cuda()
            thlist = torch.linspace(0, 1 - 1e-10, num).cuda()
        else:
            prec, recall = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
        return prec, recall

    def _S_object(self, pred, gt):
        fg = torch.where(gt == 0, torch.zeros_like(pred), pred)
        bg = torch.where(gt == 1, torch.zeros_like(pred), 1 - pred)
        o_fg = self._object(fg, gt)
        o_bg = self._object(bg, 1 - gt)
        u = gt.mean()
        Q = u * o_fg + (1 - u) * o_bg
        return Q

    def _object(self, pred, gt):
        temp = pred[gt == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

        return score

    def _S_region(self, pred, gt):
        X, Y = self._centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = self._divideGT(gt, X, Y)
        p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
        Q1 = self._ssim(p1, gt1)
        Q2 = self._ssim(p2, gt2)
        Q3 = self._ssim(p3, gt3)
        Q4 = self._ssim(p4, gt4)
        Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
        # print(Q)
        return Q

    def _centroid(self, gt):
        rows, cols = gt.size()[-2:]
        gt = gt.view(rows, cols)
        if gt.sum() == 0:
            if self.cuda:
                X = torch.eye(1).cuda() * round(cols / 2)
                Y = torch.eye(1).cuda() * round(rows / 2)
            else:
                X = torch.eye(1) * round(cols / 2)
                Y = torch.eye(1) * round(rows / 2)
        else:
            total = gt.sum()
            if self.cuda:
                i = torch.from_numpy(np.arange(0, cols)).cuda().float()
                j = torch.from_numpy(np.arange(0, rows)).cuda().float()
            else:
                i = torch.from_numpy(np.arange(0, cols)).float()
                j = torch.from_numpy(np.arange(0, rows)).float()
            X = torch.round((gt.sum(dim=0) * i).sum() / total)
            Y = torch.round((gt.sum(dim=1) * j).sum() / total)
        return X.long(), Y.long()

    def _divideGT(self, gt, X, Y):
        h, w = gt.size()[-2:]
        area = h * w
        gt = gt.view(h, w)
        LT = gt[:Y, :X]
        RT = gt[:Y, X:w]
        LB = gt[Y:h, :X]
        RB = gt[Y:h, X:w]
        X = X.float()
        Y = Y.float()
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(self, pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]
        return LT, RT, LB, RB

    def _ssim(self, pred, gt):
        gt = gt.float()
        h, w = pred.size()[-2:]
        N = h * w
        x = pred.mean()
        y = gt.mean()
        sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((gt - y) * (gt - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x) * (gt - y)).sum() / (N - 1 + 1e-20)

        aplha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-20)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0
        return Q

if __name__ == "__main__":
    from eval_dataloader import EvalDataset
    #LFSD
    # dataset='LFSD'
    # DLGRG GFRNet  LFTransNet  MEANet  MoLF
    # method='DLGRG'
    # method='GFRNet'
    # method='LFTransNet'
    # method='MEANet'
    # method='MoLF'
    # eval_path=f'/media/h335/E/shuqi/zhiying/显著性图/{method}/LFSD'
    # method='ours'
    # eval_path='/media/h335/E/shuqi/zhiying/lf_new/TMP/test256_LFSD'
    # gt_path='/media/h335/E/shuqi/zhiying/lf_data/LFSD/TestSet/gt'

    #HFUT
    # dataset='HFUT'
    # DLGRG GFRNet  LFTransNet  MEANet  MoLF
    # method='DLGRG'
    # method='GFRNet'
    # method='LFTransNet'
    # method='MEANet'
    # method='MoLF'
    # eval_path=f'/media/h335/E/shuqi/zhiying/显著性图/{method}/HFUT'
    # method='ours'
    # eval_path='/media/h335/E/shuqi/zhiying/lf_new/TMP/test256_HFUT'
    # gt_path='/media/h335/E/shuqi/zhiying/lf_data/HFUT/TestSet/gt'

    #NIPS
    dataset='NIPS'
    # DLGRG GFRNet  LFTransNet  MEANet  MoLF
    # method='DLGRG'
    # method='GFRNet'
    # method='LFTransNet'
    # method='MEANet'
    # method='MoLF'
    # eval_path=f'/media/h335/E/shuqi/zhiying/显著性图/{method}/DUTLF-FS'
    method='ours'
    eval_path='/media/h335/E/shuqi/zhiying/lf_new/TMP/Fuse256_seg_stackdiff_crossdiff_rfb_convlstm_seg_crossdiff_seghead'
    gt_path='/media/h335/E/shuqi/zhiying/lf_data/NIPS/TestSet/gt'
    
    loader = EvalDataset(eval_path, gt_path)
    # thread = Eval_thread(loader, opt.savename, 'LFSD', opt.log_path, False, epoch=epoch,txt=opt.logname)
    thread=Eval_thread(loader, method=method, dataset=dataset, output_dir='/media/h335/E/shuqi/zhiying/lf_new/log_zhiying', cuda=False, epoch=0, txt=f'{method}.txt')
    print(thread.run())