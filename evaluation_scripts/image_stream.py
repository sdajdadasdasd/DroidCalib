# Copyright (c) 2023 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# This source code is derived from DROID-SLAM (https://github.com/princeton-vl/DROID-SLAM)
# Copyright (c) 2021, Princeton Vision & Learning Lab, licensed under the BSD 3-Clause License,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

import torch.nn.functional as F
import numpy as np
import random
import torch
from droid import Droid
from types import SimpleNamespace


def image_stream(datapath, original_intr, rectified_intr, original_size, 
                rectified_size, crop=[0, 0], num_images=None, stride=1, start_index=0,
                camera_model='pinhole', intr_error=0, seed=0, undistort=True, **kwargs):

    """
    图像流生成器，用于从指定路径读取图像并进行预处理（去畸变、缩放、裁剪等），同时根据相机模型调整内参。

    参数:
        datapath (str): 包含PNG格式图像的文件夹路径
        original_intr (list or array-like): 原始相机内参，前四个元素是 [fx, fy, cx, cy]，后续可能包括畸变系数
        rectified_intr (list or array-like): 校正后的图像尺寸对应的理想内参（未使用）
        original_size (tuple of int): 原始图像尺寸 (width, height)
        rectified_size (tuple of int): 目标校正后图像尺寸 (width, height)
        crop (list of int, optional): 裁剪边缘大小 [crop_width, crop_height]，默认为 [0, 0]
        num_images (int, optional): 最多读取的图像数量，默认为 None 表示读取所有图像
        stride (int, optional): 图像采样步长，默认为 1 表示连续读取
        start_index (int, optional): 起始图像索引，默认为 0
        camera_model (str, optional): 相机模型类型，可选 'pinhole'、'mei' 或 'radial'，默认为 'pinhole'
        intr_error (float, optional): 内参误差因子，0表示无误差，默认为 0
        seed (int, optional): 随机种子，用于控制内参误差的生成，默认为 0
        undistort (bool, optional): 是否进行图像去畸变处理，默认为 True
        **kwargs: 其他关键字参数

    生成:
        tuple: 包含以下元素的元组
            - stride*t (int): 当前图像在原始序列中的时间索引
            - image[None] (torch.Tensor): 处理后的图像张量，形状为 (1, 3, H, W)
            - intr (torch.Tensor): 调整后的相机内参张量，在GPU上
            - size_factor (list): 尺寸调整因子 [width_factor, height_factor]
    """

    # 提取原始相机内参
    fx, fy, cx, cy = original_intr[:4]
    K_l = np.array([fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]).reshape(3,3)
    d_l = np.array(original_intr[4:])

    # 计算尺寸调整因子
    size_factor = [(rectified_size[0]+2*crop[0])/original_size[0], 
                   (rectified_size[1]+2*crop[1])/original_size[1]] # [w, h]

    # 读取文件夹中所有png图像并按索引、步长和数量进行筛选
    images_list = sorted(glob.glob(os.path.join(datapath, '*.png')))[start_index::stride][:num_images]

    for t, imfile in enumerate(images_list):

        # 读取、校正、调整大小和裁剪图像
        image = cv2.imread(imfile)

        if undistort:
            image = cv2.undistort(image, K_l, d_l)

        image = cv2.resize(image, (rectified_size[0]+2*crop[0], 
                                   rectified_size[1]+2*crop[1]))
        image = torch.from_numpy(image).permute(2,0,1)
        image = image[:, crop[1]:-crop[1] or None, crop[0]:-crop[0] or None] # crop if crop != 0
        
        # 根据尺寸变化调整相机内参
        fxa = fx * size_factor[0]
        fya = fy * size_factor[1]
        cxa = cx * size_factor[0]-crop[0]
        cya = cy * size_factor[1]-crop[1]

        # 根据相机模型设置不同的内参格式
        if camera_model == 'pinhole':
            intr = np.array([fxa, fya, cxa, cya])
        elif camera_model == 'mei':
            intr = np.array([fxa, fya, cxa, cya, 0.0])
        elif camera_model == 'radial':
            intr = np.array([fxa, fya, cxa, cya, 0., 0.])
        else:
            raise Exception("Camera model not implemented!")

        # 设置无信息的初始内参值（当intr_error为None时）
        if intr_error is None:
            h = rectified_size[1]
            w = rectified_size[0]
            intr[:4] = np.array([(h+w)/2, (h+w)/2, w/2, h/2])

        # 添加内参误差（当intr_error不为None时）
        else:
            np.random.seed(seed)
            errors = -intr_error + 2 * intr_error * \
                    np.random.uniform(size=intr.shape)
            intr += intr * errors

        intr = torch.as_tensor(intr).cuda()

        yield stride*t, image[None], intr, size_factor
