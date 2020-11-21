# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Demo code

@author: Denis Tome'

"""
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from base import SetType
import dataset.transform as trsf
from dataset import Mocap
from utils import config, ConsoleLogger
from utils import evaluate, utils_io
import argparse
import os
from model import resnet as pose_resnet
# from model import pose_resnet
from model import encoder_decoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.vis3d import show3Dpose
from utils.vis2d import draw2Dpred_and_gt
import cv2
import pprint
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import yaml
from easydict import EasyDict as edict

import pdb

LOGGER = ConsoleLogger("Demo", 'test')

def parse_args():
    parser = argparse.ArgumentParser(description="demo script")
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--load_model', type=str)
    parser.add_argument('--data', default='test', type=str)  # "train", "val", "test"
    args = parser.parse_args()

    return args


def main():
    """Main"""
    args = parse_args()
    print('Starting demo...')
    device = torch.device(f"cuda:{args.gpu}")
    LOGGER.info((args))

    # ------------------- Data loader -------------------

    data_transform = transforms.Compose([
        trsf.ImageTrsf(),  # normalize
        trsf.Joints3DTrsf(),  # centerize
        trsf.ToTensor()])  # to tensor

    data = Mocap(
        config.dataset[args.data],
        SetType.TEST,
        transform=data_transform)
    data_loader = DataLoader(
        data,
        batch_size=16,
        shuffle=config.data_loader.shuffle,
        num_workers=8)

    # ------------------- Evaluation -------------------

    eval_body = evaluate.EvalBody()
    eval_upper = evaluate.EvalUpperBody()
    eval_lower = evaluate.EvalLowerBody()

    # ------------------- Model -------------------
    with open('model/model.yaml') as fin:
        model_cfg = edict(yaml.safe_load(fin))
    resnet = pose_resnet.get_pose_net(model_cfg, False)
    # resnet = pose_resnet.get_pose_net(False)
    autoencoder = encoder_decoder.AutoEncoder()

    if args.load_model:
        if not os.path.isfile(args.load_model):
            raise ValueError(f"No checkpoint found at {args.load_model}")
        checkpoint = torch.load(args.load_model, map_location=device)
        resnet.load_state_dict(checkpoint['resnet_state_dict'])
        autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
    else:
        raise ValueError("No checkpoint!")

    resnet.cuda(device)
    autoencoder.cuda(device)
    resnet.eval()
    autoencoder.eval()

    # ------------------- Read dataset frames -------------------
    fig = plt.figure(figsize=(19.2, 10.8))
    plt.axis('off')
    subplot_idx = 1

    with torch.no_grad():
        for it, (img, p2d, p3d, heatmap, action) in enumerate(data_loader):

            print('Iteration: {}'.format(it))
            print('Images: {}'.format(img.shape))
            print('p2ds: {}'.format(p2d.shape))
            print('p3ds: {}'.format(p3d.shape))
            print('Actions: {}'.format(action))

            img = img.to(device)
            p3d = p3d.to(device)
            # heatmap = heatmap.to(device)

            heatmap2d_hat = resnet(img)  # torch.Size([16, 15, 48, 48])
            p3d_hat, heatmap2d_recon = autoencoder(heatmap2d_hat)

            # Evaluate results using different evaluation metrices
            y_output = p3d_hat.data.cpu().numpy()
            y_target = p3d.data.cpu().numpy()

            eval_body.eval(y_output, y_target, action)
            eval_upper.eval(y_output, y_target, action)
            eval_lower.eval(y_output, y_target, action)


            # ------------------- Visualize 3D pose -------------------
            if subplot_idx <= 32:
                ax1 = fig.add_subplot(4, 8, subplot_idx, projection='3d')
                show3Dpose(p3d[0].cpu().numpy(), ax1, True)

                # Plot 3d gt
                # ax2 = fig.add_subplot(4, 8, subplot_idx+1, projection='3d')
                show3Dpose(p3d_hat[0].detach().cpu().numpy(), ax1, False)

                subplot_idx += 1
            if subplot_idx == 33:
                plt.savefig(os.path.join(LOGGER.logfile_dir, 'vis.png'))

            # ------------------- Visualize 2D heatmap -------------------

            if it < 32:

                # gt
                img_grid = draw2Dpred_and_gt(img, heatmap, (368,368))  # tensor
                img_grid = img_grid.numpy().transpose(1, 2, 0)
                cv2.imwrite(os.path.join(LOGGER.logfile_dir, f'gt_{it}.jpg'), img_grid)

                # 2d reconstruction
                img_grid_recon = draw2Dpred_and_gt(img, heatmap2d_recon, (368, 368), p2d.clone())
                img_grid_recon = img_grid_recon.numpy().transpose(1, 2, 0)
                cv2.imwrite(os.path.join(LOGGER.logfile_dir, f"recon_{it}.jpg"), img_grid_recon)

                # 2d prediction
                img_grid_hat = draw2Dpred_and_gt(img, heatmap2d_hat, (368,368), p2d.clone())  # tensor
                img_grid_hat = img_grid_hat.numpy().transpose(1,2,0)
                cv2.imwrite(os.path.join(LOGGER.logfile_dir, f'pred_{it}.jpg'), img_grid_hat)





    # ------------------- Save results -------------------

    LOGGER.info('Saving evaluation results...')
    res = {'FullBody': eval_body.get_results(),
           'UpperBody': eval_upper.get_results(),
           'LowerBody': eval_lower.get_results()}

    LOGGER.info(pprint.pformat(res))
    print('Done.')


if __name__ == "__main__":
    main()
