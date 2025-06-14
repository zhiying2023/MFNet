import torch
import torch.nn as nn
import argparse
import os.path as osp
import os
from eval_evaluator import Eval_thread
from eval_dataloader import EvalDataset


# from concurrent.futures import ThreadPoolExecutor
def main(cfg):
    root_dir = cfg.root_dir
    if cfg.save_dir is not None:
        output_dir = cfg.save_dir
    else:
        output_dir = root_dir
    gt_dir = osp.join(root_dir, 'gt')
    pred_dir = osp.join(root_dir, 'report1') # pred # 1//2//3//4//5
    if cfg.methods is None:
        method_names = os.listdir(pred_dir)
    else:
        method_names = cfg.methods.split(' ')
    if cfg.datasets is None:
        dataset_names = os.listdir(gt_dir)
    else:
        dataset_names = cfg.datasets.split(' ')

    threads = []
    for method in method_names:
        for dataset in dataset_names:
            loader = EvalDataset(osp.join(pred_dir, method, dataset), osp.join(gt_dir, dataset))
            thread = Eval_thread(loader, method, dataset, output_dir, cfg.cuda)
            threads.append(thread)
    for thread in threads:
        print(thread.run())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--methods', type=str, default=None)
    # parser.add_argument('--datasets', type=str, default=None)
    # parser.add_argument('--root_dir', type=str, default='/home/hanqi/test/')
    # parser.add_argument('--save_dir', type=str, default=None)
    # parser.add_argument('--cuda', type=bool, default=True)

    parser.add_argument('--methods', type=str, default=None)
    parser.add_argument('--datasets', type=str, default='fs')
    parser.add_argument('--root_dir', type=str, default='./')
    parser.add_argument('--save_dir', type=str, default='./')
    parser.add_argument('--cuda', type=bool, default=False)

    config = parser.parse_args()
    main(config)
