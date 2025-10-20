# Copyright (c) 2023 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0
#
# This source code is derived from DROID-SLAM (https://github.com/princeton-vl/DROID-SLAM)
# Copyright (c) 2021, Princeton Vision & Learning Lab, licensed under the BSD 3-Clause License,
# cf. 3rd-party-licenses.txt file in the root directory of this source tree.

import torch
import torch.nn.functional as F

from lietorch import SE3, Sim3

MIN_DEPTH = 0.2

def extract_intrinsics(intrinsics):
    """
    提取相机内参参数
    
    参数:
        intrinsics: 相机内参张量，通常包含 [fx, fy, cx, cy] 或更多参数
        
    返回:
        分解后的内参参数元组
    """
    return intrinsics[...,None,None,:].unbind(dim=-1)

def coords_grid(ht, wd, **kwargs):
    """
    创建坐标网格
    
    参数:
        ht: 网格高度
        wd: 网格宽度
        **kwargs: 其他参数，如设备类型
        
    返回:
        坐标网格张量
    """
    y, x = torch.meshgrid(
        torch.arange(ht).to(**kwargs).float(),
        torch.arange(wd).to(**kwargs).float())

    return torch.stack([x, y], dim=-1)

def iproj(disps, intrinsics, jacobian=False):
    """
    针孔相机逆投影函数
    
    将图像上的点和对应的深度值逆投影到3D空间中。
    核心公式:
    对于图像点 (u, v) 和深度 d，对应的3D点为:
    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d
    
    参数:
        disps: 视差图/深度图
        intrinsics: 相机内参 [fx, fy, cx, cy]
        jacobian: 是否计算雅可比矩阵
        
    返回:
        pts: 3D点坐标
        J: 雅可比矩阵（如果需要）
    """
    ht, wd = disps.shape[2:]
    fx, fy, cx, cy = extract_intrinsics(intrinsics)
    
    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float())

    i = torch.ones_like(disps)
    # 逆投影公式: 将像素坐标转换为归一化相机坐标
    X = (x - cx) / fx
    Y = (y - cy) / fy
    pts = torch.stack([X, Y, i, disps], dim=-1)

    if jacobian:
        J = torch.zeros_like(pts)
        J[...,-1] = 1.0
        return pts, J

    return pts, None

def proj(Xs, intrinsics, jacobian=False, return_depth=False):
    """
    针孔相机投影函数
    
    将3D空间中的点投影到图像平面上。
    核心公式:
    对于3D点 (X, Y, Z)，对应的图像点为:
    u = fx * X / Z + cx
    v = fy * Y / Z + cy
    
    参数:
        Xs: 3D点坐标 [X, Y, Z, D]
        intrinsics: 相机内参 [fx, fy, cx, cy]
        jacobian: 是否计算雅可比矩阵
        return_depth: 是否返回深度信息
        
    返回:
        coords: 投影后的图像坐标
        proj_jac: 投影的雅可比矩阵（如果需要）
    """
    fx, fy, cx, cy = extract_intrinsics(intrinsics)
    X, Y, Z, D = Xs.unbind(dim=-1)

    # 防止深度值过小导致数值不稳定
    Z = torch.where(Z < 0.5*MIN_DEPTH, torch.ones_like(Z), Z)
    d = 1.0 / Z

    # 投影公式: 将3D点投影到图像平面
    x = fx * (X * d) + cx
    y = fy * (Y * d) + cy
    if return_depth:
        coords = torch.stack([x, y, D*d], dim=-1)
    else:
        coords = torch.stack([x, y], dim=-1)

    if jacobian:
        B, N, H, W = d.shape
        o = torch.zeros_like(d)
        # 计算投影函数的雅可比矩阵
        # 表示图像坐标对3D点坐标的偏导数
        proj_jac = torch.stack([
             fx*d,     o, -fx*X*d*d,  o,
                o,  fy*d, -fy*Y*d*d,  o,
                # o,     o,    -D*d*d,  d,
        ], dim=-1).view(B, N, H, W, 2, 4)

        return coords, proj_jac

    return coords, None

def actp(Gij, X0, jacobian=False):
    """
    对点云执行变换操作
    
    使用李群变换矩阵 Gij 对点云 X0 进行变换。
    核心公式:
    X1 = Gij * X0
    
    参数:
        Gij: 变换矩阵（李群表示）
        X0: 原始点云
        jacobian: 是否计算雅可比矩阵
        
    返回:
        X1: 变换后的点云
        Ja: 变换的雅可比矩阵（如果需要）
    """
    X1 = Gij[:,:,None,None] * X0
    
    if jacobian:
        X, Y, Z, d = X1.unbind(dim=-1)
        o = torch.zeros_like(d)
        B, N, H, W = d.shape

        if isinstance(Gij, SE3):
            # SE3变换的雅可比矩阵
            # 表示变换后的点对李代数参数的偏导数
            Ja = torch.stack([
                d,  o,  o,  o,  Z, -Y,
                o,  d,  o, -Z,  o,  X, 
                o,  o,  d,  Y, -X,  o,
                o,  o,  o,  o,  o,  o,
            ], dim=-1).view(B, N, H, W, 4, 6)

        elif isinstance(Gij, Sim3):
            Ja = torch.stack([
                d,  o,  o,  o,  Z, -Y,  X,
                o,  d,  o, -Z,  o,  X,  Y,
                o,  o,  d,  Y, -X,  o,  Z,
                o,  o,  o,  o,  o,  o,  o
            ], dim=-1).view(B, N, H, W, 4, 7)

        return X1, Ja

    return X1, None

def projective_transform(poses, depths, intrinsics, ii, jj, jacobian=False, return_depth=False):
    """
    投影变换函数，将点从第ii帧映射到第jj帧
    
    这是视觉SLAM中关键的几何操作，包含三个步骤:
    1. 逆投影: 将图像点逆投影到3D空间
    2. 变换: 使用相机位姿变换3D点
    3. 投影: 将3D点投影到目标图像平面
    
    参数:
        poses: 相机位姿
        depths: 深度图
        intrinsics: 相机内参
        ii: 源帧索引
        jj: 目标帧索引
        jacobian: 是否计算雅可比矩阵
        return_depth: 是否返回深度信息
        
    返回:
        x1: 投影后的坐标
        valid: 有效点掩码
        (Ji, Jj, Jz): 雅可比矩阵（如果需要）
    """
    # 逆投影 (针孔相机模型)
    X0, Jz = iproj(depths[:,ii], intrinsics[:,ii], jacobian=jacobian)
    
    # 计算帧间变换矩阵
    Gij = poses[:,jj] * poses[:,ii].inv()

    # 处理相同帧的情况
    Gij.data[:,ii==jj] = torch.as_tensor([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device="cuda")
    X1, Ja = actp(Gij, X0, jacobian=jacobian)
    
    # 投影 (针孔相机模型)
    x1, Jp = proj(X1, intrinsics[:,jj], jacobian=jacobian, return_depth=return_depth)

    # 排除太靠近相机的点
    valid = ((X1[...,2] > MIN_DEPTH) & (X0[...,2] > MIN_DEPTH)).float()
    valid = valid.unsqueeze(-1)

    if jacobian:
        # 计算复合变换的雅可比矩阵
        # Ji 根据对偶伴随变换
        Jj = torch.matmul(Jp, Ja)
        Ji = -Gij[:,:,None,None,None].adjT(Jj)

        Jz = Gij[:,:,None,None] * Jz
        Jz = torch.matmul(Jp, Jz.unsqueeze(-1))

        return x1, valid, (Ji, Jj, Jz)

    return x1, valid

def induced_flow(poses, disps, intrinsics, ii, jj):
    """
    由相机运动引起的光流计算
    
    参数:
        poses: 相机位姿
        disps: 视差图
        intrinsics: 相机内参
        ii: 源帧索引
        jj: 目标帧索引
        
    返回:
        光流场和有效性掩码
    """

    ht, wd = disps.shape[2:]
    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float())

    coords0 = torch.stack([x, y], dim=-1)
    coords1, valid = projective_transform(poses, disps, intrinsics, ii, jj, False)

    return coords1[...,:2] - coords0, valid


def general_projective_transform(poses, depths, intr, ii, jj, jacobian=False, 
                                 return_depth=False, model_id=0):
    """
    通用投影变换函数，支持多种相机模型
    
    参数:
        poses: 相机位姿
        depths: 深度图
        intr: 相机内参
        ii: 源帧索引
        jj: 目标帧索引
        jacobian: 是否计算雅可比矩阵
        return_depth: 是否返回深度信息
        model_id: 相机模型ID (0: 针孔, 1: MEI, 2: 焦距)
    """
    
    if (model_id == 0) or (model_id == 2): # pinhole or focal
        return projective_transform(poses, depths, intr, ii, jj, jacobian, return_depth)
    
    elif model_id == 1: # mei
        return projective_transform_mei(poses, depths, intr, ii, jj, jacobian, return_depth)
    else:
        raise Exception('Camera model not implemented.')


def projective_transform_mei(poses, depths, intr, ii, jj, jacobian=False, return_depth=False):
    """
    MEI相机模型的投影变换
    
    参数:
        poses: 相机位姿
        depths: 深度图
        intr: MEI相机内参 [fx, fy, cx, cy, xi]
        ii: 源帧索引
        jj: 目标帧索引
        jacobian: 是否计算雅可比矩阵
        return_depth: 是否返回深度信息
    """

    """ map points from ii->jj """
    if torch.sum(torch.isnan(depths))>0:
        raise Exception('nan values in depth')
    
    # 逆投影
    X0, _, _ = iproj_mei(depths[:,ii], intr[:,ii], jacobian=jacobian)

    # 变换
    Gij = poses[:,jj] * poses[:,ii].inv()

    Gij.data[:,ii==jj] = torch.as_tensor([-0.1, 0.0, 0.0, 0.0, 0.0, 
                                            0.0, 1.0], device="cuda")
    X1, _ = actp(Gij, X0, jacobian=jacobian)
    
    # 投影
    x1, _, _ = proj_mei(X1, intr[:,jj], jacobian=jacobian, 
                                return_depth=return_depth)

    # 排除太靠近相机的点
    valid = ((X1[...,2] > MIN_DEPTH) & (X0[...,2] > MIN_DEPTH)).float()
    valid = valid.unsqueeze(-1)

    if jacobian:
        raise Exception("Jacobian for mei model currently not supported.")

    return x1, valid


def iproj_mei(disps, intr, jacobian=False):
    """
    MEI相机逆投影函数
    
    MEI相机模型（统一相机模型）的逆投影实现。
    核心公式:
    r̂ = ((x - cx) / fx)^2 + ((y - cy) / fy)^2
    factor = (ξ + √(1 + (1 - ξ^2) * r̂)) / (1 + r̂)
    X = (x - cx) * factor / fx
    Y = (y - cy) * factor / fy
    Z = factor - ξ
    
    参数:
        disps: 视差图
        intr: MEI相机内参 [fx, fy, cx, cy, xi]
        jacobian: 是否计算雅可比矩阵
        
    返回:
        pts: 3D点坐标
        None, None: 占位符
    """
    ht, wd = disps.shape[2:]
    fx, fy, cx, cy, xi = extract_intrinsics(intr)
    
    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float())

    # 计算MEI模型的逆投影因子
    rhat = ((x - cx) / fx)**2 + ((y - cy) / fy)**2
    factor = (xi + torch.sqrt(1 + (1 - xi**2) * rhat)) / (1 + rhat)

    # MEI逆投影公式
    X = (x - cx) * factor / fx
    Y = (y - cy) * factor / fy
    Z = factor - xi

    pts = torch.stack([X/Z, Y/Z, Z/Z, disps], dim=-1)

    if jacobian:
        raise Exception("Jacobian for mei model currently not supported.")

    return pts, None, None


def proj_mei(Xs, intr, jacobian=False, return_depth=False):
    """
    MEI相机投影函数
    
    MEI相机模型（统一相机模型）的投影实现。
    核心公式:
    r = √(X^2 + Y^2 + Z^2)
    factor = 1 / (Z + ξ * r)
    u = fx * X * factor + cx
    v = fy * Y * factor + cy
    
    参数:
        Xs: 3D点坐标 [X, Y, Z, D]
        intr: MEI相机内参 [fx, fy, cx, cy, xi]
        jacobian: 是否计算雅可比矩阵
        return_depth: 是否返回深度信息
        
    返回:
        coords: 投影后的图像坐标
        None, None: 占位符
    """
    fx, fy, cx, cy, xi = extract_intrinsics(intr)
    X, Y, Z, D = Xs.unbind(dim=-1)

    # 防止深度值过小
    Z = torch.where(Z < 0.5*MIN_DEPTH, torch.ones_like(Z), Z)

    d = 1.0 / Z
    # 计算MEI模型的投影因子
    r = torch.sqrt(X**2 + Y**2 + Z**2)
    factor = 1.0 / (Z + xi * r)

    # MEI投影公式
    x = fx * (X * factor) + cx
    y = fy * (Y * factor) + cy

    if return_depth:
        coords = torch.stack([x, y, D*d], dim=-1)
    else:
        coords = torch.stack([x, y], dim=-1)

    if jacobian:
        raise Exception("Jacobian for mei model currently not supported.")

    return coords, None, None
