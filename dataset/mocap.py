# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# -*- coding: utf-8 -*-
"""
Data processing where only Images and associated 3D
joint positions are loaded.

@author: Denis Tome'

"""
import os
from skimage import io as sio
import numpy as np
from base import BaseDataset
from utils import utils_io, config
from utils.joint2heatmap import joint2heatmap


class Mocap(BaseDataset):
    """Mocap Dataset loader"""

    ROOT_DIRS = ['rgba', 'json']
    CM_TO_M = 100

    def index_db(self):

        return self._index_dir(self.path)  # data/Dataset/TrainSet

    def _index_dir(self, path):
        """Recursively add paths to the set of
        indexed files

        Arguments:
            path {str} -- folder path

        Returns:
            dict -- indexed files per root dir
        """

        indexed_paths = dict()
        sub_dirs, _ = utils_io.get_subdirs(path)

        # if this is the last level of directory
        if set(self.ROOT_DIRS) <= set(sub_dirs):

            # get files from subdirs
            n_frames = -1

            # let's extract the rgba and json data per frame
            for sub_dir in self.ROOT_DIRS:
                d_path = os.path.join(path, sub_dir)
                _, paths = utils_io.get_files(d_path)

                if n_frames < 0:
                    n_frames = len(paths)
                else:
                    if len(paths) != n_frames:
                        raise ValueError(
                            'Frames info in {} not matching other passes'.format(d_path))

                encoded = [p.encode('utf8') for p in paths]
                indexed_paths.update({sub_dir: encoded})

            return indexed_paths

        # initialize indexed_paths
        for sub_dir in self.ROOT_DIRS:
            indexed_paths.update({sub_dir: []})

        # check subdirs of path and merge info
        for sub_dir in sub_dirs:
            indexed = self._index_dir(os.path.join(path, sub_dir))

            for r_dir in self.ROOT_DIRS:
                indexed_paths[r_dir].extend(indexed[r_dir])

        return indexed_paths

    def _process_points(self, data):
        """Filter joints to select only a sub-set for
        training/evaluation

        Arguments:
            data {dict} -- data dictionary with frame info

        Returns:
            np.ndarray -- 2D joint positƒions, format (J x 2)
            np.ndarray -- 3D joint positions, format (J x 3)
        """

        p2d_orig = np.array(data['pts2d_fisheye']).T
        p3d_orig = np.array(data['pts3d_fisheye']).T
        joint_names = {j['name'].replace('mixamorig:', ''): jid
                       for jid, j in enumerate(data['joints'])}

        # ------------------- Filter joints -------------------

        p2d = np.empty([len(config.skel), 2], dtype=p2d_orig.dtype)
        p3d = np.empty([len(config.skel), 3], dtype=p2d_orig.dtype)

        for jid, j in enumerate(config.skel.keys()):
            p2d[jid] = p2d_orig[joint_names[j]]
            p3d[jid] = p3d_orig[joint_names[j]]

        p3d /= self.CM_TO_M

        return p2d, p3d

    def __getitem__(self, index):

        # load image
        img_path = self.index['rgba'][index].decode('utf8')
        img = sio.imread(img_path)

        # (800, 1280, 3) -> (800, 800, 3)
        start = int((img.shape[1]-img.shape[0])/2)
        img = img[:, start:start+img.shape[0], :]

        # read joint positions
        json_path = self.index['json'][index].decode('utf8')
        data = utils_io.read_json(json_path)
        p2d, p3d = self._process_points(data)

        # p2d rescale according to img
        start = 240
        p2d[:, 0] -= start  # x-axis
        p2d[:, 0] /= (800 / config.data.image_size[1])
        p2d[:, 1] /= (800 / config.data.image_size[0])

        heatmap, _ = joint2heatmap(p2d)

        # get action name
        action = data['action']

        if self.transform:
            img = self.transform({'image': img})['image']
            p3d = self.transform({'joints3D': p3d})['joints3D']
            p2d = self.transform({'joints2D': p2d})['joints2D']
            heatmap = self.transform({'heatmap': heatmap})['heatmap']
        # only use 15 heatmaps (w/o head)
        heatmap = np.delete(heatmap, 1, 0)

        return img, p2d, p3d, heatmap, action

    def __len__(self):

        return len(self.index[self.ROOT_DIRS[0]])
