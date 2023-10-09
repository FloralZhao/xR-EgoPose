from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import torch.backends.cudnn as cudnn

from base import SetType
import dataset.transform as trsf
from dataset import Mocap
from utils import config, ConsoleLogger
from utils import evaluate, utils_io
from utils import arguments
import torch.optim as optim
from model import resnet as pose_resnet
from model import encoder_decoder
import itertools
import torch.nn as nn
from utils.loss import HeatmapLoss
import pprint
import os
from tensorboardX import SummaryWriter
from utils import AverageMeter
import time
from easydict import EasyDict as edict
import yaml

import pdb


def main():
    """Main"""

    args = arguments.parse_args()
    LOGGER = ConsoleLogger(args.training_type, 'train')
    logdir = LOGGER.getLogFolder()
    LOGGER.info(args)
    LOGGER.info(config)


    cudnn.benckmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    # ------------------- Data loader -------------------

    data_transform = transforms.Compose([
        trsf.ImageTrsf(),  # normalize
        trsf.Joints3DTrsf(),  # centerize
        trsf.ToTensor()])  # to tensor

    # training data
    train_data = Mocap(
        config.dataset.train,
        SetType.TRAIN,
        transform=data_transform)
    train_data_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=config.data_loader.shuffle,
        num_workers=8)

    val_data = Mocap(
        config.dataset.val,
        SetType.VAL,
        transform=data_transform)
    val_data_loader = DataLoader(
        val_data,
        batch_size=2,
        shuffle=config.data_loader.shuffle,
        num_workers=8)

    # ------------------- Model -------------------
    if args.training_type != 'Train3d':
        with open('model/model.yaml') as fin:
            model_cfg = edict(yaml.safe_load(fin))
        resnet = pose_resnet.get_pose_net(model_cfg, True)
        Loss2D = HeatmapLoss()  # same as MSELoss()
        # LossMSE = nn.MSELoss()
    if args.training_type != 'Train2d':
        autoencoder = encoder_decoder.AutoEncoder()

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        if args.training_type != 'Train3d':
            resnet = resnet.cuda(device)
            Loss2D = Loss2D.cuda(device)
        if args.training_type != 'Train2d':
            autoencoder = autoencoder.cuda(device)

    # ------------------- optimizer -------------------
    if args.training_type == 'Train2d':
        optimizer = optim.Adam(resnet.parameters(), lr=args.learning_rate)
    if args.training_type == 'Train3d':
        optimizer = optim.Adam(autoencoder.parameters(), lr=config.train.learning_rate)
    if args.training_type != 'Finetune':
        optimizer = optim.Adam(itertools.chain(resnet.parameters(), autoencoder.parameters()), lr=config.train.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    # ------------------- load model -------------------
    if args.load_model:
        if not os.path.isfile(args.load_model):
            raise ValueError(f"No checkpoint found at {args.load_model}")
        checkpoint = torch.load(args.load_model)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if args.training_type != 'Train3d':
            resnet.load_state_dict(checkpoint['resnet_state_dict'])
        if args.training_type != 'Train2d':
            autoencoder.load_state_dict(checkpoint['autoencoder_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler'])


    # ------------------- tensorboard -------------------
    train_global_steps = 0
    writer_dict = {
        'writer': SummaryWriter(log_dir=logdir),
        'train_global_steps': train_global_steps
    }

    # ------------------- Evaluation -------------------
    if args.training_type != 'Train2d':
        eval_body = evaluate.EvalBody()
        eval_upper = evaluate.EvalUpperBody()
        eval_lower = evaluate.EvalLowerBody()


    best_perf = float('inf')
    best_model = False
    # ------------------- run the model -------------------
    for epoch in range(args.epochs):
        with torch.autograd.set_detect_anomaly(True):
            LOGGER.info(f'====Training epoch {epoch}====')
            losses = AverageMeter()
            batch_time = AverageMeter()

            resnet.train()
            autoencoder.train()

            end = time.time()
            for it, (img, p2d, p3d, heatmap, action) in enumerate(train_data_loader, 0):

                img = img.to(device)
                p2d = p2d.to(device)
                p3d = p3d.to(device)
                heatmap = heatmap.to(device)

                if args.training_type != 'Train3d':
                    heatmap2d_hat = resnet(img)  # torch.Size([16, 15, 48, 48])
                else:
                    heatmap2d_hat = heatmap
                p3d_hat, heatmap2d_recon = autoencoder(heatmap2d_hat)

                loss2d = Loss2D(heatmap, heatmap2d_hat).mean()
                # loss2d = LossMSE(heatmap, heatmap2d_hat)

                if args.training_type == 'Train2d':
                    loss = loss2d
                elif args.training_type == 'Train3d':
                    pass

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_time.update(time.time() - end)
                losses.update(loss.item(), img.size(0))

                if it % config.train.PRINT_FREQ == 0:
                    # logging messages
                    msg = 'Epoch: [{0}][{1}/{2}]\t' \
                          'Batch Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                          'Speed {speed:.1f} samples/s\t' \
                          'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                        epoch, it, len(train_data_loader), batch_time=batch_time,
                        speed=img.size(0) / batch_time.val,  # averaged within batch
                        loss=losses)
                    LOGGER.info(msg)
                end = time.time()
            scheduler.step()

            # ------------------- validation -------------------

            resnet.eval()
            autoencoder.eval()

            if args.training_type != 'Train2d':
                # Evaluate results using different evaluation metrices
                y_output = p3d_hat.data.cpu().numpy()
                y_target = p3d.data.cpu().numpy()

                eval_body.eval(y_output, y_target, action)
                eval_upper.eval(y_output, y_target, action)
                eval_lower.eval(y_output, y_target, action)


            # ------------------- Save results -------------------
            checkpoint_dir = os.path.join(logdir, 'checkpoints')
            LOGGER.info('=> saving checkpoint to {}'.format(checkpoint_dir))
            states = dict()
            if args.training_type!='Train3d':
                states['resnet_state_dict'] = resnet.state_dict()
            if args.training_type!='Train2d':
                states['autoencoder_state_dict'] = autoencoder.state_dict()
            states['optimizer_state_dict']= optimizer.state_dict()

            torch.save(states, f'checkpoint_{epoch}.tar')
            res = {'FullBody': eval_body.get_results(),
                   'UpperBody': eval_upper.get_results(),
                   'LowerBody': eval_lower.get_results()}

            utils_io.write_json(config.eval.output_file, res)

            LOGGER.info('Done.')


if __name__ == "__main__":
    main()
