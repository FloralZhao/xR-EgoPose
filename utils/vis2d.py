import cv2
import skimage.io as sio
from skimage.transform import resize as resize
import numpy as np
import json
from . import config
# from config import config
import torchvision, torch
import pdb

num_joints = 16
output_size = (368, 368, 3)

I = np.array([0, 2, 3, 5, 6, 8, 9, 10, 12, 13, 14, 2, 5, 2])  # start points
J = np.array([1, 3, 4, 6, 7, 9, 10, 11, 13, 14, 15, 8, 12, 5])  # end points

def drawBones(img, p2d, img_name=None):
    assert len(p2d.shape)==2 and p2d.shape[0]==num_joints and p2d.shape[1]==2, f"invalid p2d shape {p2d.shape}"

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
        img = cv2.circle(img, tuple(p2d[i]), 3, (0, 0, 255), thickness=3)
    for i in range(len(I)):
        img = cv2.line(img, tuple(p2d[I[i]]), tuple(p2d[J[i]]), (255, 0, 0), thickness=2)

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


def draw2Dpred_and_gt(img, heatmaps, output_size=(48,48)):
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
    img = resize(img, output_size, anti_aliasing=True)  # (48, 48, 3) float64
    heatmap = heatmaps[0].detach().cpu().numpy()  # (15, 48, 48)
    image_to_show = torch.rand(heatmap.shape[0], 3, output_size[0], output_size[1])
    img *= 255
    img.astype(np.uint8)
    for i in range(heatmap.shape[0]):
        ht = resize(heatmap[i], output_size)
        image_to_show[i] = torch.from_numpy(colorize(ht).transpose((2,0,1))) * 0.5 + torch.from_numpy(img.transpose((2, 0, 1))) * 0.5

    img_grid = torchvision.utils.make_grid(image_to_show, nrow=4).to(torch.uint8)
    return img_grid





if __name__ == "__main__":
    img = '/data/i/dongxu/xR-EgoPose/data/Dataset/TrainSet/female_001_a_a/env_001/cam_down/rgba/female_001_a_a.rgba.003688.png'
    with open('/data/i/dongxu/xR-EgoPose/data/Dataset/TrainSet/female_001_a_a/env_001/cam_down/json/female_001_a_a_003688.json') as file:
        data = json.load(file)

    p2d_orig = np.array(data['pts2d_fisheye']).T
    joint_names = {j['name'].replace('mixamorig:', ''): jid
                   for jid, j in enumerate(data['joints'])}

    p2d = np.empty([num_joints, 2], dtype=p2d_orig.dtype)
    for jid, j in enumerate(config.skel.keys()):
        p2d[jid] = p2d_orig[joint_names[j]]

    img_vis = drawBones(img, p2d)
    # TODO: change RGB to BGR
    cv2.imwrite('female_001_a_a.rgba.003688.png', img_vis)
    cv2.imshow('t', img_vis)
    cv2.waitKey()


