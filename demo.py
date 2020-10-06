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
from model import pose_resnet, encoder_decoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.vis3d import show3Dpose
from mpl_toolkits.mplot3d import Axes3D

import pdb

LOGGER = ConsoleLogger("Demo", 'test')

def parse_args():
    parser = argparse.ArgumentParser(description="Training script")
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
    resnet = pose_resnet.get_pose_net(True)
    autoencoder = encoder_decoder.AutoEncoder()
    resnet.cuda(device)
    autoencoder.cuda(device)
    if args.load_model:
        if not os.path.isfile(args.load_model):
            raise ValueError(f"No checkpoint found at {args.load_model}")
        checkpoint = torch.load(args.load_model)
        resnet.load_state_dict(checkpoint['resnet_state_dict'])
        autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])


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
            p3d_hat, _ = autoencoder(heatmap2d_hat)

            # Evaluate results using different evaluation metrices
            y_output = p3d_hat.data.cpu().numpy()
            y_target = p3d.data.cpu().numpy()

            eval_body.eval(y_output, y_target, action)
            eval_upper.eval(y_output, y_target, action)
            eval_lower.eval(y_output, y_target, action)

            if subplot_idx <= 32:
                ax1 = fig.add_subplot(4, 8, subplot_idx, projection='3d')
                show3Dpose(p3d[0].cpu().numpy(), ax1, True)

                # Plot 3d gt
                ax2 = fig.add_subplot(4, 8, subplot_idx+1, projection='3d')
                show3Dpose(p3d_hat[0].detach().cpu().numpy(), ax2, False)

                subplot_idx += 2
            if subplot_idx == 33:
                plt.savefig(os.path.join(LOGGER.logfile_dir, 'vis.png'))
                break



    # ------------------- Save results -------------------

    LOGGER.info('Saving evaluation results...')
    res = {'FullBody': eval_body.get_results(),
           'UpperBody': eval_upper.get_results(),
           'LowerBody': eval_lower.get_results()}

    LOGGER.info(res)
    print('Done.')


if __name__ == "__main__":
    main()
