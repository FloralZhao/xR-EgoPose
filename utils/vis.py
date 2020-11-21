import cv2
import skimage.io as sio
from skimage.transform import resize as resize
import numpy as np
import json
from . import config
# from config import config
import torchvision, torch
import pdb
import math

num_joints = 16
output_size = (368, 368, 3)

I = np.array([0, 2, 3, 5, 6, 8, 9, 10, 12, 13, 14, 2, 5, 2])  # start points
J = np.array([1, 3, 4, 6, 7, 9, 10, 11, 13, 14, 15, 8, 12, 5])  # end points

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
    start = int((img.shape[1] - img.shape[0]) / 2)
    img = img[:, start:start + img.shape[0], :]

    p2d[:, 0] -= start  # x-axis
    p2d[:, 0] /= (img.shape[0] / output_size[0])
    p2d[:, 1] /= (img.shape[1] / output_size[1])
    p2d = p2d.astype(np.int)

    # skimage functions will convert dtype from uint8 to float64.
    img = resize(img, output_size, anti_aliasing=True)  # convert to scale [0, 1]
    img*=255
    # Convert to uint8. Otherwise pixel >= 1 is white
    img = img.astype(np.uint8)

    for i in range(p2d.shape[0]):
        img = cv2.circle(img, tuple(p2d[i]), 3, color, thickness=3)
    for i in range(len(I)):
        img = cv2.line(img, tuple(p2d[I[i]]), tuple(p2d[J[i]]), color, thickness=2)

    if img_name is not None:
        cv2.imwrite(img_name, img)


    return img

