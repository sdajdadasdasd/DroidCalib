# Copyright (c) 2023 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

from scipy.optimize import least_squares
import numpy as np

N_GRIDPOINTS = 1000

def proj_pinhole(points, fx, fy, ppx, ppy):
    """
    针孔相机模型的投影函数
    
    核心公式:
    u = fx * X / Z + ppx
    v = fy * Y / Z + ppy
    
    参数:
        points: 3D点坐标 [3, N]，N为点的数量
        fx, fy: 焦距
        ppx, ppy: 主点坐标
        
    返回:
        point_proj: 投影后的图像点坐标 [2, N]
    """
    npoints = points.shape[1]
    point_proj = np.zeros((2, npoints))
    # 归一化坐标: 将3D点投影到归一化图像平面
    points_normalized = points[0:2, :] / points[2, :]
    # 应用内参: 从归一化坐标转换到像素坐标
    point_proj[0, :] = points_normalized[0, :] * fx + ppx
    point_proj[1, :] = points_normalized[1, :] * fy + ppy
    return point_proj

def iproj_pinhole(uv, fx, fy, ppx, ppy):
    """
    针孔相机模型的逆投影函数
    
    核心公式:
    X = (u - ppx) / fx
    Y = (v - ppy) / fy
    Z = 1
    
    参数:
        uv: 图像点坐标 [2, N]
        fx, fy: 焦距
        ppx, ppy: 主点坐标
        
    返回:
        rays: 逆投影后的射线方向 [3, N]
    """
    npoints = uv.shape[1]
    rays = np.zeros((3, npoints))
    # 从像素坐标转换到归一化坐标
    rays[0, :] = (uv[0, :] - ppx) / fx
    rays[1, :] = (uv[1, :] - ppy) / fy
    # 设置Z坐标为1，得到归一化相机坐标系下的射线方向
    rays[2, :] = 1
    return rays

def proj_radial(points, fx, fy, ppx, ppy, k1, k2):
    """
    径向畸变相机模型的投影函数
    
    核心公式:
    n = (X/Z)^2 + (Y/Z)^2
    r = 1 + k1 * n + k2 * n^2
    u = fx * (X/Z) * r + ppx
    v = fy * (Y/Z) * r + ppy
    
    参数:
        points: 3D点坐标 [3, N]
        fx, fy: 焦距
        ppx, ppy: 主点坐标
        k1, k2: 径向畸变系数
        
    返回:
        point_proj: 投影后的图像点坐标 [2, N]
    """
    npoints = points.shape[1]
    point_proj = np.zeros((2, npoints))
    # 归一化坐标
    points_normalized = points[0:2, :] / points[2, :]
    # 计算径向距离的平方
    n = np.sum(points_normalized ** 2, axis=0)
    # 径向畸变因子
    r = 1 + k1 * n + k2 * n ** 2
    # 应用径向畸变和内参
    point_proj[0, :] = points_normalized[0, :] * fx * r + ppx
    point_proj[1, :] = points_normalized[1, :] * fy * r + ppy
    return point_proj



def proj_mei(points, fx, fy, ppx, ppy, xi, k1=0, k2=0, k3=0):
    """
    MEI（统一）相机模型的投影函数
    
    MEI模型是能够同时处理普通相机和鱼眼相机的统一模型。
    核心公式:
    分子分母部分:
    r = sqrt(X^2 + Y^2 + Z^2)
    denominator = Z + xi * r
    
    归一化坐标:
    x_n = X / denominator
    y_n = Y / denominator
    
    径向畸变:
    r2 = x_n^2 + y_n^2
    m = 1 + k1*r2 + k2*r2^2 + k3*r2^3
    xd_n = m * x_n
    yd_n = m * y_n
    
    像素坐标:
    u = fx * xd_n + ppx
    v = fy * yd_n + ppy
    
    参数:
        points: 3D点坐标 [3, N]
        fx, fy: 焦距
        ppx, ppy: 主点坐标
        xi: 鱼眼相机参数
        k1, k2, k3: 径向畸变系数
        
    返回:
        point_proj: 投影后的图像点坐标 [2, N]
    """
    npoints = points.shape[1]
    point_proj = np.zeros((2, npoints))
    # 计算归一化坐标（MEI模型特有）
    points_normalized = points[0:2, :] / (points[2, :] + xi * 
                        np.sqrt(np.sum(points ** 2, axis=0)))

    # 计算径向畸变因子
    r2 = np.sum(points_normalized ** 2, axis=0)
    m = 1 + k1 * r2 + k2 * np.power(r2, 2) + k3 * np.power(r2, 3)

    # 应用径向畸变
    points_distorted = np.zeros((2, npoints))
    points_distorted[0, :] = m * points_normalized[0, :]
    points_distorted[1, :] = m * points_normalized[1, :]

    # 应用内参转换到像素坐标
    point_proj[0, :] = points_distorted[0, :] * fx + ppx
    point_proj[1, :] = points_distorted[1, :] * fy + ppy

    return point_proj


def iproj_mei(uv, fx, fy, ppx, ppy, xi):
    """
    MEI（统一）相机模型的逆投影函数
    
    核心公式:
    逆向求解过程，从像素坐标计算射线方向。
    
    参数:
        uv: 图像点坐标 [2, N]
        fx, fy: 焦距
        ppx, ppy: 主点坐标
        xi: 鱼眼相机参数
        
    返回:
        rays: 逆投影后的射线方向 [3, N]
    """
    npoints = uv.shape[1]
    nXY = np.zeros((3, npoints))
    # 从像素坐标转换到归一化坐标
    nXY[0, :] = (uv[0, :] - ppx) / fx
    nXY[1, :] = (uv[1, :] - ppy) / fy

    # MEI模型逆投影的核心计算
    rays = np.zeros((3, npoints))
    rays[2, :] = (xi + np.sqrt(1 + (1 - xi**2) * (nXY[0, :]**2 + 
                 nXY[1, :]**2))) / (1 + nXY[0, :]**2 + nXY[1, :]**2) - xi
    rays[0, :] = nXY[0, :] * (rays[2, :] + xi)
    rays[1, :] = nXY[1, :] * (rays[2, :] + xi)
    return rays


def rotate_rodrigues(points, rot_vecs):
    """
    使用罗德里格斯公式旋转点
    
    罗德里格斯旋转公式是将旋转轴和旋转角度表示的旋转转换为旋转矩阵的标准方法。
    核心公式:
    v = rotation_axis / ||rotation_axis||  # 单位旋转轴
    theta = ||rotation_axis||  # 旋转角度
    R = I + sin(theta)*[v]× + (1-cos(theta))*[v]×^2  # 旋转矩阵
    rotated_points = R * points
    
    其中 [v]× 是向量v的反对称矩阵。
    
    参数:
        points: 待旋转的点 [3, N]
        rot_vecs: 旋转轴和角度的组合表示 [3] 或 [N, 3]
        
    返回:
        旋转后的点
    """
    """Rotate points by given rotation vectors.
    Rodrigues' rotation formula is used.
    """
    if rot_vecs.ndim > 1:
        theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]

    else:
        theta = np.linalg.norm(rot_vecs)
        rot_vecs = np.array([rot_vecs for _ in range(points.shape[1])]).T

    if theta == 0:
        return points

    with np.errstate(divide='ignore', invalid='ignore'):
        v = np.true_divide(rot_vecs, theta)
        v[v == np.inf] = 0
        v = np.nan_to_num(v)

    # 计算点积项
    dot = np.sum(points * v, axis=0)[np.newaxis, :]
    # 计算三角函数值
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    # 应用罗德里格斯公式
    return cos_theta * points + sin_theta * np.cross(v, points, axisa=0, axisb=0).T + dot * (1 - cos_theta) * v


def proj_general(points, intr):
    """
    通用投影函数，根据内参数量自动选择对应的投影模型
    
    参数:
        points: 3D点坐标
        intr: 相机内参
        
    返回:
        投影后的图像点坐标
    """
    if intr.shape[0] == 4:
        points_proj = proj_pinhole(points, *intr)
    elif intr.shape[0] == 5:
        points_proj = proj_mei(points, *intr)
    elif intr.shape[0] == 6:
        points_proj = proj_radial(points, *intr)
    return points_proj

def iproj_general(points, intr):
    """
    通用逆投影函数，根据内参数量自动选择对应的逆投影模型
    
    参数:
        points: 图像点坐标
        intr: 相机内参
        
    返回:
        逆投影后的射线方向
    """
    if intr.shape[0] == 4:
        points_proj = iproj_pinhole(points, *intr)
    elif intr.shape[0] == 5:
        points_proj = iproj_mei(points, *intr)
    elif intr.shape[0] == 6:
        points_proj = iproj_radial(points, *intr)
    return points_proj


def mapping_error(intr_a, intr_b, image_size):
    """
    计算两个相机模型之间的映射误差
    
    方法概述:
    1. 生成覆盖整个图像的2D图像点
    2. 使用最优参数intr_a投影这些点得到射线
    3. 使用待比较参数intr_b将射线重新投影到图像空间
    4. 计算两个模型之间的残差
    
    重要说明:
    内参的变化可以通过外参的变化进行补偿。因此，我们必须计算经过外参调整补偿后的最小可能偏差。
    即：在调整外参后计算最小的相机模型偏差。
    参见 Hagemann et al. IJCV 2021
    
    参数:
        intr_a: 参考相机内参
        intr_b: 待比较相机内参
        image_size: 图像尺寸 [width, height]
        
    返回:
        rms_error: 均方根误差
    """

    def cost(r, intr_a, intr_b, image_size):
        """
        优化目标函数
        
        参数:
            r: 补偿旋转参数
            intr_a: 参考相机内参
            intr_b: 待比较相机内参
            image_size: 图像尺寸
            
        返回:
            residuals: 残差向量
        """

        nx = int(np.sqrt(N_GRIDPOINTS))
        ny = int(np.sqrt(N_GRIDPOINTS))
        # 在图像上生成均匀分布的网格点
        x = np.linspace(0, image_size[0], nx+4)[2:-2]
        y = np.linspace(0, image_size[1], ny+4)[2:-2]
        points = [[xi, yi, 1] for xi in x for yi in y]
        points = np.array(points).T

        # 如果参数相同且旋转为0，则残差为0
        if len(intr_a) == len(intr_b) and (intr_a == intr_b).all() and np.sum(r) == 0:
            residuals = np.zeros(2 * nx * ny)
            return residuals

        # 逆投影得到射线
        rays = iproj_general(points, intr_a)
        # 应用补偿旋转
        X_cam_opt = rotate_rodrigues(rays, r)
        # 使用待比较参数重新投影
        points_proj = proj_general(X_cam_opt, intr_b)

        # 计算残差
        residuals = np.zeros(2 * nx * ny)
        residuals[::2] = points[0, :] - points_proj[0, :]
        residuals[1::2] = points[1, :] - points_proj[1, :]

        return residuals
    
    # 初始化补偿旋转参数
    compensating_rotation = np.zeros(3)
    # 使用最小二乘法优化补偿旋转参数
    res = least_squares(cost, compensating_rotation, 
                        args=(intr_a, intr_b, image_size))
    residuals = res.fun
    # 计算均方根误差
    return np.sqrt(np.mean(residuals**2))
