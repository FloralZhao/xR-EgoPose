import numpy as np
import torch
import torch.nn as nn


class HeatmapLoss(nn.Module):
    """
    loss for detection heatmap (stacked hourglass model)
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        l = ((pred - gt)**2)
        l = l.mean(dim=-1).mean(dim=-1).mean(dim=-1)
        return l ## l of dim bsize



class HeatmapLossSquare(nn.Module):
    """
        loss for detection heatmap (stacked hourglass model)
        not mse loss (use 'sum' instead of 'mean')
        """

    def __init__(self):
        super(HeatmapLossSquare, self).__init__()

    def forward(self, pred, gt):
        l = ((pred - gt) ** 2)
        l = l.sum(dim=-1).sum(dim=-1).sum(dim=-1)
        return l  ## l of dim bsize


class PoseLoss(nn.Module):
    '''
    MPJPE
    '''
    def __init__(self):
        super(PoseLoss, self).__init__()
    def forward(self, pred, gt):
        l = (pred - gt) ** 2
        l = torch.sqrt(torch.sum(l, dim=-1)).mean(-1)
        return l  ## l of dim bsize


class LimbLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.start_point = np.array([0, 2, 3, 5, 6, 8, 9, 10, 12, 13, 14, 2, 5, 2])  # start points
        self.end_point = np.array([1, 3, 4, 6, 7, 9, 10, 11, 13, 14, 15, 8, 12, 5])  # end points

    def forward(self, pred, gt):
        p, p_hat = self.getBones(pred, gt)  #(16, 14, 3)
        CosineSimilarity = nn.CosineSimilarity(dim=2)  # (16, 14)
        theta = CosineSimilarity(p, p_hat).sum(dim=-1)  # (16)

        R = torch.norm(p-p_hat, dim=-1) # (16, 14)
        R = R.sum(dim=-1)  # (16)

        return theta, R

    def getBones(self, pred, gt):
        batch_size = pred.size(0)
        limb_num = self.start_point.shape[0]
        limb = torch.zeros(batch_size, limb_num, 3).to(gt.device)
        limb_hat = torch.zeros(batch_size, limb_num, 3).to(pred.device)
        for i in range(limb_num):
            limb[:, i, :] = gt[:, self.end_point[i], :] - gt[:, self.start_point[i], :]
            limb_hat[:, i, :] = pred[:, self.end_point[i], :] - pred[:, self.start_point[i], :]

        return limb, limb_hat


