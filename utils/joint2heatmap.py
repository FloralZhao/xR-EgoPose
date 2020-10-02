import numpy as np
from . import config
# from config import config
import json
import cv2
import skimage.io as sio
from skimage.transform import resize as resize
import os
import pdb


heatmap_size = (int(config.data.heatmap_size[0]), int(config.data.heatmap_size[1]))
# heatmap_size = (368, 368)


def joint2heatmeap(p2d, sigma=2, heatmap_type='gaussian'):
    '''
    Args:
        joints: [num_joints, 3]
        sigma: std for gaussian
        heatmap_type

    Returns:
        visible(1: visible, 0: not visible)

    '''
    num_joints = len(config.skel)
    visible = np.ones((num_joints, 1), dtype=np.float32)

    assert heatmap_type == 'gaussian', 'Only support gaussian map now!'

    start = 240
    p2d[:, 0] -= start  # x-axis
    p2d[:, 0] /= (800 / heatmap_size[1])
    p2d[:, 1] /= (800 / heatmap_size[0])
    p2d = p2d.astype(np.int)


    if heatmap_type == 'gaussian':
        heatmaps = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]), dtype=np.float32)
        tmp_size = sigma*3


        for joint in range(num_joints):
            # feat_stride = config.data.image_size / config.data.heatmap_size
            # mu_x = int(p2d[joint][0] / feat_stride[0] + 0.5)
            # mu_y = int(p2d[joint][1] / feat_stride[1] + 0.5)
            mu_x = p2d[joint][0]
            mu_y = p2d[joint][1]

            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

            if ul[0] >= heatmap_size[1] or ul[1] >= heatmap_size[0] or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                visible[joint] = 0
                continue

            # generate Gaussian
            sz = 2 * tmp_size + 1
            x = np.arange(0, sz, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = sz // 2
            # The gaussian is not normalized, we want the cneter value to equal 1
            g = np.exp(-((x-x0)**2 + (y-y0)**2) / (2*sigma**2))

            # usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], heatmap_size[1]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_size[0]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], heatmap_size[1])
            img_y = max(0, ul[1]), min(br[1], heatmap_size[0])

            heatmaps[joint][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return heatmaps, visible


if __name__ == "__main__":
    img = '/data/i/dongxu/xR-EgoPose/data/Dataset/TrainSet/female_001_a_a/env_001/cam_down/rgba/female_001_a_a.rgba.003688.png'
    with open('/data/i/dongxu/xR-EgoPose/data/Dataset/TrainSet/female_001_a_a/env_001/cam_down/json/female_001_a_a_003688.json') as file:
        data = json.load(file)

    p2d_orig = np.array(data['pts2d_fisheye']).T
    joint_names = {j['name'].replace('mixamorig:', ''): jid
                   for jid, j in enumerate(data['joints'])}

    p2d = np.empty([16, 2], dtype=p2d_orig.dtype)
    for jid, j in enumerate(config.skel.keys()):
        p2d[jid] = p2d_orig[joint_names[j]]

    heatmaps, _ = joint2heatmeap(p2d)
    # TODO: change RGB to BGR
    img = sio.imread(img)
    start = int((img.shape[1] - img.shape[0]) / 2)
    img = img[:, start:start + img.shape[0], :]
    img = resize(img, heatmap_size, anti_aliasing=True)
    for i in range(15):

        img_fuse = 0.5*img[:,:,0] + heatmaps[i]*0.5
        img_fuse *= 255
        # pdb.set_trace()
        cv2.imwrite(f"test_imgs/test_{i}.png", img_fuse)

        # cv2.imshow('t', img_fuse)
        # cv2.waitKey()








