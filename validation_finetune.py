# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Demo code

@author: Denis Tome'

"""
from base import SetType
import dataset.transform as trsf
from dataset import Mocap
from utils import config, ConsoleLogger, AverageMeter
from utils import evaluate, utils_io
import os
import pdb
from utils.loss import HeatmapLoss
import pprint
import torch



def validate(LOGGER, data_loader, resnet, autoencoder, device, epoch):


    # ------------------- Loss -------------------


    # ------------------- Evaluation -------------------
    eval_body = evaluate.EvalBody()
    eval_upper = evaluate.EvalUpperBody()
    eval_lower = evaluate.EvalLowerBody()

    # ------------------- validate -------------------
    with torch.no_grad():
        for it, (img, p2d, p3d, heatmap, action) in enumerate(data_loader):
            img = img.to(device)
            p3d = p3d.to(device)

            heatmap2d_hat = resnet(img)
            p3d_hat, _ = autoencoder(heatmap2d_hat)

            # Evaluate results using different evaluation metrices
            y_output = p3d_hat.data.cpu().numpy()
            y_target = p3d.data.cpu().numpy()

            eval_body.eval(y_output, y_target, action)
            eval_upper.eval(y_output, y_target, action)
            eval_lower.eval(y_output, y_target, action)


        # ------------------- Save results -------------------

        LOGGER.info('===========Evaluation on Val data==========')
        res = {'FullBody': eval_body.get_results(),
               'UpperBody': eval_upper.get_results(),
               'LowerBody': eval_lower.get_results()}
        LOGGER.info(pprint.pformat(res))

        # utils_io.write_json(os.path.join(LOGGER.logfile_dir, f'eval_val_{epoch}'+'.json'), res)

    return eval_body.get_results()['All']


