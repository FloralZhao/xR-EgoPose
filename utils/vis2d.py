import cv2
import skimage.io as sio
from skimage.transform import resize as resize
import numpy as np
import json
from . import config
import torchvision, torch
import pdb
import math
from torchvision import transforms as transforms
from torch.utils.data import DataLoader
from base.base_dataset import SetType
from dataset import Mocap
import dataset.transform as trsf
import os

num_joints = 15
CM_TO_M = 100

I = np.array([1, 2, 4, 5, 7, 8, 9, 11, 12, 13, 1, 4, 1])  # start points
J = np.array([2, 3, 5, 6, 8, 9, 10, 12, 13, 14, 7, 11, 4])  # end points

def drawBones(img, p2d, gt, img_name=None):
    '''
    Args:
        img: str (filename) or np.array
        p2d:
        gt:
        img_name:

    Returns:

    '''
    assert len(p2d.shape)==2 and p2d.shape[0]==num_joints and p2d.shape[1]==2, f"invalid p2d shape {p2d.shape}"

    if gt:
        color = (0, 0, 255)
    else:
        color = (255, 0, 0)
    if isinstance(img, str):
        img = sio.imread(img)

    p2d = p2d.astype(np.int)

    # Convert to uint8. Otherwise pixel >= 1 is white
    img = img.astype(np.uint8)

    for i in range(p2d.shape[0]):
        img = cv2.circle(img, tuple(p2d[i]), 3, color, thickness=3)
    for i in range(len(I)):
        img = cv2.line(img, tuple(p2d[I[i]]), tuple(p2d[J[i]]), color, thickness=2)

    if img_name is not None:
        cv2.imwrite(img_name, img)


    return img


# =============== from CPM pytorch  =====================
def showHeatmap(h, joint):
    ind = config.skel[joint]['jid']
    h = h[0]  # （15， 48， 48）
    h = h.cpu().detach().numpy()
    h = h[ind] * 255
    h = np.uint8(np.clip(h, 0, 255))
    cv2.imshow('heatmap', h)
    cv2.waitKey()

def getJetColor(v, vmin, vmax):
    c = np.zeros((3))
    if (v < vmin):
        v = vmin
    if (v > vmax):
        v = vmax
    dv = vmax - vmin
    if (v < (vmin + 0.125 * dv)):
        c[0] = 256 * (0.5 + (v * 4)) #B: 0.5 ~ 1
    elif (v < (vmin + 0.375 * dv)):
        c[0] = 255
        c[1] = 256 * (v - 0.125) * 4 #G: 0 ~ 1
    elif (v < (vmin + 0.625 * dv)):
        c[0] = 256 * (-4 * v + 2.5)  #B: 1 ~ 0
        c[1] = 255
        c[2] = 256 * (4 * (v - 0.375)) #R: 0 ~ 1
    elif (v < (vmin + 0.875 * dv)):
        c[1] = 256 * (-4 * v + 3.5)  #G: 1 ~ 0
        c[2] = 255
    else:
        c[2] = 256 * (-4 * v + 4.5) #R: 1 ~ 0.5
    return c

def colorize(gray_img):
    out = np.zeros(gray_img.shape + (3,))
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y,x,:] = getJetColor(gray_img[y,x], 0, 1)
    return out


def draw2Dpred_and_gt(img, heatmaps, output_size=(48,48), pose_2d=None):
    '''
    Args:
        img: (16, 3, 368, 368), torch.Tensor
        heatmaps: (16, 15, 48, 48), torch.Tensor

    Returns:

    '''

    # only visualize first sample
    img = img[0]  # (3, 368, 368) torch.Tensor
    img = img.cpu().numpy().transpose(1,2,0)  # (368, 368, 3) numpy.array, float32
    img = img * 0.5 + 0.5
    # img = img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]
    img = resize(img, output_size, anti_aliasing=True)  # (output_size, 3) float64
    img *= 255
    # img.astype(np.uint8)
    heatmap = heatmaps[0].detach().cpu().numpy()  # (15, 48, 48)

    image_to_show = torch.rand(heatmap.shape[0]+1, 3, output_size[0], output_size[1])

    for i in range(heatmap.shape[0]):
        ht = resize(heatmap[i], output_size)  # np.array, float
        image_to_show[i] = torch.from_numpy(colorize(ht).transpose((2,0,1))) * 0.3 + torch.from_numpy(img.transpose((2, 0, 1))) * 0.7

    if pose_2d is not None:
        pose_2d = pose_2d.numpy()
        image_to_show[-1] = torch.from_numpy((visualize_joints(img.astype(np.uint8), heatmap, pose_2d[0], output_size)).transpose(2,0,1))
    img_grid = torchvision.utils.make_grid(image_to_show, nrow=4).to(torch.uint8)
    return img_grid


# =============== from PoseResnet  =====================

def get_max_preds(batch_heatmaps):
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray), \
        'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def save_batch_heatmaps(batch_image, batch_heatmaps, file_name=None, output_size=(368,368),
                        normalize=True):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    '''
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros((batch_size*output_size[0],
                           (num_joints+1)*output_size[1],
                           3),
                          dtype=np.uint8)

    preds, maxvals = get_max_preds(batch_heatmaps.detach().cpu().numpy())
    preds[:,:,0] *= (output_size[1]/heatmap_width)
    preds[:,:,1] *= (output_size[0]/heatmap_height)



    for i in range(batch_size):
        image = batch_image[i].mul(255)\
                              .clamp(0, 255)\
                              .byte()\
                              .permute(1, 2, 0)\
                              .cpu().numpy()
        heatmaps = batch_heatmaps[i].mul(255)\
                                    .clamp(0, 255)\
                                    .byte()\
                                    .cpu().numpy()

        resized_image = cv2.resize(image,
                                   output_size)

        height_begin = output_size[0] * i
        height_end = output_size[0] * (i + 1)
        for j in range(num_joints):
            cv2.circle(resized_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       5, [0, 0, 255], 5)
            heatmap = cv2.resize(heatmaps[j, :, :], output_size)

            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap*0.7 + resized_image*0.3
            cv2.circle(masked_image,
                       (int(preds[i][j][0]), int(preds[i][j][1])),
                       1, [0, 0, 255], 1)

            width_begin = output_size[1] * (j+1)
            width_end = output_size[1] * (j+2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = \
                masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3
        grid_image[height_begin:height_end, 0:output_size[1], :] = resized_image
    if file_name is not None:
        cv2.imwrite(file_name, grid_image)
    return grid_image


def save_batch_image_with_joints(batch_image, batch_joints, batch_joints_vis,
                                 file_name, nrow=8, padding=2):
    '''
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_joints, 3],
    batch_joints_vis: [batch_size, num_joints, 1],
    }
    '''
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)  # batch_size
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            joints = batch_joints[k]
            joints_vis = batch_joints_vis[k]

            for joint, joint_vis in zip(joints, joints_vis):
                joint[0] = x * width + padding + joint[0]
                joint[1] = y * height + padding + joint[1]
                if joint_vis[0]:
                    cv2.circle(ndarr, (int(joint[0]), int(joint[1])), 2, [255, 0, 0], 2)
            k = k + 1
    if file_name is not None:
        cv2.imwrite(file_name, ndarr)
    return ndarr



def visualize_joints(img, heatmap_hat, pose_gt, output_size):
    '''

    Args:
        img: numpy.array (height, width, 3) uint8
        heatmap_hat: torch.tensor (num_joints, height, width)
        pose_gt: numpy array (15, 2)

    Returns:

    '''

    heatmap_height = heatmap_hat.shape[1]
    heatmap_width = heatmap_hat.shape[2]

    preds, maxvals = get_max_preds(np.expand_dims(heatmap_hat, 0))
    preds = preds[0]
    preds[:, 0] *= (output_size[1] / heatmap_width)
    preds[:, 1] *= (output_size[0] / heatmap_height)


    pose_gt[:, 0] *= (output_size[1] / config.data.image_size[1])
    pose_gt[:, 1] *= (output_size[0] / config.data.image_size[0])
    # pose_gt[:, 0] *= (output_size[1] / heatmap_width)
    # pose_gt[:, 1] *= (output_size[0] / heatmap_height)

    pose_gt = np.delete(pose_gt, 1, 0)

    img = drawBones(img, preds, False)
    img = drawBones(img, pose_gt, True, img_name=None)

    return img



if __name__ == "__main__":

    folder = '/data/i/dongxu/xR-EgoPose/test_dir'
    if not os.path.exists(folder):
        os.makedirs(folder)
    data_transform = transforms.Compose([
        trsf.ImageTrsf(),  # normalize
        trsf.Joints3DTrsf(),  # centerize
        trsf.ToTensor()])  # to tensor

    train_data = Mocap(
        config.dataset.train,
        SetType.TRAIN,
        transform=data_transform)
    train_data_loader = DataLoader(
        train_data,
        batch_size=4,
        shuffle=False,
        num_workers=1)

    for it, (img, p2d, p3d, heatmap, action) in enumerate(train_data_loader, 0):
        img_grid = draw2Dpred_and_gt(img, heatmap, (48, 48), p2d)
        cv2.imwrite(os.path.join(folder, f"test_{it}.jpg"), img_grid.numpy().transpose((1,2,0)))



