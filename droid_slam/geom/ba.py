import lietorch
import torch
import torch.nn.functional as F

from .chol import block_solve, schur_solve
import geom.projective_ops as pops

from torch_scatter import scatter_sum


# utility functions for scattering ops
def safe_scatter_add_mat(A, ii, jj, n, m):
    """
    安全地执行矩阵元素的散射加法操作
    
    参数:
        A: 输入张量
        ii: 行索引
        jj: 列索引
        n: 行数
        m: 列数
    """
    v = (ii >= 0) & (jj >= 0) & (ii < n) & (jj < m)
    return scatter_sum(A[:,v], ii[v]*m + jj[v], dim=1, dim_size=n*m)

def safe_scatter_add_vec(b, ii, n):
    """
    安全地执行向量元素的散射加法操作
    
    参数:
        b: 输入向量
        ii: 索引
        n: 向量长度
    """
    v = (ii >= 0) & (ii < n)
    return scatter_sum(b[:,v], ii[v], dim=1, dim_size=n)

# apply retraction operator to inv-depth maps
def disp_retr(disps, dz, ii):
    """
    对逆深度图应用回退操作（retraction）
    
    参数:
        disps: 逆深度图
        dz: 深度更新量
        ii: 索引
    """
    ii = ii.to(device=dz.device)
    return disps + scatter_sum(dz, ii, dim=1, dim_size=disps.shape[1])

# apply retraction operator to poses
def pose_retr(poses, dx, ii):
    """
    对位姿应用回退操作（retraction）
    
    参数:
        poses: 位姿
        dx: 位姿更新量
        ii: 索引
    """
    ii = ii.to(device=dx.device)
    return poses.retr(scatter_sum(dx, ii, dim=1, dim_size=poses.shape[1]))


def BA(target, weight, eta, poses, disps, intrinsics, ii, jj, fixedp=1, rig=1):
    """
    全光束法平差 (Full Bundle Adjustment)
    
    该函数实现了完整的光束法平差算法，同时优化相机位姿、逆深度和相机内参。
    核心公式:
    1. 残差计算: r = target - coords
    2. 加权最小二乘: min ||W^(1/2) * r||^2
    3. 线性化系统: H*dx = -b
    4. 舒尔补求解: 通过消元法求解位姿和深度的更新量

    参数:
        target: 目标坐标 (图像上的观测点)
        weight: 权重矩阵
        eta: 正则化参数
        poses: 相机位姿
        disps: 逆深度图
        intrinsics: 相机内参
        ii: 关键帧索引
        jj: 目标帧索引
        fixedp: 固定位姿数
        rig: rig参数

    返回:
        poses: 更新后的相机位姿
        disps: 更新后的逆深度图
    """

    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim

    ### 1: 计算雅可比矩阵和残差 ###
    # 通过投影变换计算坐标和雅可比矩阵
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True)

    # 计算残差向量: 观测值与预测值的差
    r = (target - coords).view(B, N, -1, 1)
    # 应用权重和有效性掩码
    w = .001 * (valid * weight).view(B, N, -1, 1)

    ### 2: 构建线性系统 ###
    # 重塑雅可比矩阵维度
    Ji = Ji.reshape(B, N, -1, D)
    Jj = Jj.reshape(B, N, -1, D)
    # 计算加权雅可比矩阵转置
    wJiT = (w * Ji).transpose(2,3)
    wJjT = (w * Jj).transpose(2,3)

    # 重塑深度雅可比矩阵
    Jz = Jz.reshape(B, N, ht*wd, -1)

    # 构建海塞矩阵块 (Hessian matrix blocks)
    # Hessian矩阵表示目标函数的二阶导数信息
    Hii = torch.matmul(wJiT, Ji)  # 对角块: 相对于第i帧位姿的二阶导数
    Hij = torch.matmul(wJiT, Jj)  # 非对角块: 相对于第i帧和第j帧位姿的二阶导数
    Hji = torch.matmul(wJjT, Ji)  # 非对角块: 相对于第j帧和第i帧位姿的二阶导数
    Hjj = torch.matmul(wJjT, Jj)  # 对角块: 相对于第j帧位姿的二阶导数

    # 构建梯度向量 (Gradient vectors)
    # 梯度向量表示目标函数的一阶导数信息
    vi = torch.matmul(wJiT, r).squeeze(-1)  # 相对于第i帧位姿的一阶导数
    vj = torch.matmul(wJjT, r).squeeze(-1)  # 相对于第j帧位姿的一阶导数

    # 构建与深度相关的项
    Ei = (wJiT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)
    Ej = (wJjT.view(B,N,D,ht*wd,-1) * Jz[:,:,None]).sum(dim=-1)

    w = w.view(B, N, ht*wd, -1)
    r = r.view(B, N, ht*wd, -1)
    # wk和Ck用于舒尔补计算中的深度相关项
    wk = torch.sum(w*r*Jz, dim=-1)  # 混合项: 位姿和深度之间的耦合
    Ck = torch.sum(w*Jz*Jz, dim=-1)  # 深度项的对角块

    # 获取唯一的关键帧索引
    kx, kk = torch.unique(ii, return_inverse=True)
    M = kx.shape[0]

    # 只优化关键帧位姿
    P = P // rig - fixedp
    ii = ii // rig - fixedp
    jj = jj // rig - fixedp

    # 使用散射加法构建完整的海塞矩阵和梯度向量
    H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hjj, jj, jj, P, P)

    E = safe_scatter_add_mat(Ei, ii, kk, P, M) + \
        safe_scatter_add_mat(Ej, jj, kk, P, M)

    v = safe_scatter_add_vec(vi, ii, P) + \
        safe_scatter_add_vec(vj, jj, P)

    C = safe_scatter_add_vec(Ck, kk, M)
    w = safe_scatter_add_vec(wk, kk, M)

    # 添加正则化项和小量以保证数值稳定性
    C = C + eta.view(*C.shape) + 1e-7

    # 重塑矩阵以适应求解器格式
    H = H.view(B, P, P, D, D)
    E = E.view(B, P, M, D, ht*wd)

    ### 3: 求解线性系统 ###
    # 使用舒尔补方法求解位姿更新量dx和深度更新量dz
    # 舒尔补是解决稀疏线性系统的一种有效方法，通过消元减少问题规模
    dx, dz = schur_solve(H, E, C, v, w)
    
    ### 4: 应用回退操作 ###
    # 将计算出的更新量应用到位姿和深度上
    poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    disps = disp_retr(disps, dz.view(B,-1,ht,wd), kx)

    # 对深度进行约束处理，避免不合理的深度值
    disps = torch.where(disps > 10, torch.zeros_like(disps), disps)
    disps = disps.clamp(min=0.0)

    return poses, disps


def MoBA(target, weight, eta, poses, disps, intrinsics, ii, jj, fixedp=1, rig=1):
    """
    仅运动光束法平差 (Motion only bundle adjustment)
    
    只优化相机位姿，不优化深度和内参的简化版本BA算法

    参数:
        target: 目标坐标
        weight: 权重矩阵
        eta: 正则化参数
        poses: 相机位姿
        disps: 逆深度图
        intrinsics: 相机内参
        ii: 关键帧索引
        jj: 目标帧索引
        fixedp: 固定位姿数
        rig: rig参数

    返回:
        poses: 更新后的相机位姿
    """

    B, P, ht, wd = disps.shape
    N = ii.shape[0]
    D = poses.manifold_dim

    ### 1: 计算雅可比矩阵和残差 ###
    coords, valid, (Ji, Jj, Jz) = pops.projective_transform(
        poses, disps, intrinsics, ii, jj, jacobian=True)

    r = (target - coords).view(B, N, -1, 1)
    w = .001 * (valid * weight).view(B, N, -1, 1)

    ### 2: 构建线性系统 ###
    Ji = Ji.reshape(B, N, -1, D)
    Jj = Jj.reshape(B, N, -1, D)
    wJiT = (w * Ji).transpose(2,3)
    wJjT = (w * Jj).transpose(2,3)

    # 构建海塞矩阵块
    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    # 构建梯度向量
    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    # 只优化关键帧位姿
    P = P // rig - fixedp
    ii = ii // rig - fixedp
    jj = jj // rig - fixedp

    # 构建完整的海塞矩阵和梯度向量
    H = safe_scatter_add_mat(Hii, ii, ii, P, P) + \
        safe_scatter_add_mat(Hij, ii, jj, P, P) + \
        safe_scatter_add_mat(Hji, jj, ii, P, P) + \
        safe_scatter_add_mat(Hjj, jj, jj, P, P)

    v = safe_scatter_add_vec(vi, ii, P) + \
        safe_scatter_add_vec(vj, jj, P)
    
    H = H.view(B, P, P, D, D)

    ### 3: 求解线性系统 ###
    # 使用块求解器直接求解位姿更新量
    dx = block_solve(H, v)

    ### 4: 应用回退操作 ###
    poses = pose_retr(poses, dx, torch.arange(P) + fixedp)
    return poses
