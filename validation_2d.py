from base import SetType
import torch
import dataset.transform as trsf
from dataset import Mocap
from utils import config, ConsoleLogger, AverageMeter
from utils import evaluate, utils_io
import os
import pdb
from utils.loss import HeatmapLoss



def validate(LOGGER, data_loader, resnet, device, epoch):

    Loss2D = HeatmapLoss()
    val_losses = AverageMeter()
    Loss2D.cuda(device)

    with torch.no_grad():
        for it, (img, p2d, p3d, heatmap, action) in enumerate(data_loader):
            img = img.to(device)
            heatmap = heatmap.to(device)


            heatmap2d_hat = resnet(img)  # torch.Size([16, 15, 48, 48])

            loss2d = Loss2D(heatmap2d_hat, heatmap).mean()

            val_losses.update(loss2d.item(), img.size(0))


        # ------------------- Save results -------------------

        LOGGER.info('Saving evaluation results...')
        msg = 'Test:\t' \
              'Loss {loss.avg:.5f}\t'.format(loss=val_losses)
        LOGGER.info(msg)

    return val_losses.avg

