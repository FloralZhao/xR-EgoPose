from base import SetType
import dataset.transform as trsf
from dataset import Mocap
from utils import config, ConsoleLogger, AverageMeter
from utils import evaluate, utils_io
import os
import pdb
from utils.loss import HeatmapLoss



def validate(LOGGER, data_loader, resnet, autoencoder, device, epoch):


    # ------------------- Loss -------------------

    Loss2D = HeatmapLoss()

    # ------------------- Evaluation -------------------
    eval_body = evaluate.EvalBody()
    eval_upper = evaluate.EvalUpperBody()
    eval_lower = evaluate.EvalLowerBody()

    # ------------------- validate -------------------
    val_losses = AverageMeter()
    for it, (img, p2d, p3d, heatmap, action) in enumerate(data_loader):
        img = img.to(device)
        p2d = p2d.to(device)
        p3d = p3d.to(device)
        heatmap = heatmap.to(device)
        Loss2D.cuda()

        heatmap2d_hat = resnet(img)  # torch.Size([16, 15, 48, 48])
        p3d_hat, heatmap2d_recon = autoencoder(heatmap2d_hat)

        loss2d = Loss2D(heatmap, heatmap2d_hat).mean()


        # Evaluate results using different evaluation metrices
        y_output = p3d_hat.data.cpu().numpy()
        y_target = p3d.data.cpu().numpy()

        eval_body.eval(y_output, y_target, action)
        eval_upper.eval(y_output, y_target, action)
        eval_lower.eval(y_output, y_target, action)


    # ------------------- Save results -------------------

    LOGGER.info('Saving evaluation results...')
    res = {'FullBody': eval_body.get_results(),
           'UpperBody': eval_upper.get_results(),
           'LowerBody': eval_lower.get_results()}
    LOGGER.info(res)
    utils_io.write_json(os.path.join(LOGGER.logfile_dir, f'eval_res_{epoch}'+'.json'), res)



