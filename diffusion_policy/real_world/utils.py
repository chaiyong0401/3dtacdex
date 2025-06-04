import torch 
from typing import Tuple
import numpy as np

def compute_quat_dist(q1, q2):
    dist = abs(q1-q2).sum()
    return dist

def transform_quat(prev_quat, cur_quat):
    cur_neg_quat = -cur_quat.copy()
    diff_pos_quat_prev = compute_quat_dist(cur_quat, prev_quat)
    diff_neg_quat_prev = compute_quat_dist(cur_neg_quat, prev_quat)
    if diff_neg_quat_prev + 1e-5 < diff_pos_quat_prev:
        cur_quat = cur_neg_quat
    return cur_quat

@torch.jit.script
def copysign(a: float, b: torch.Tensor) -> torch.Tensor:
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)

def scale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Normalizes a given input tensor to a range of [-1, 1].

    @note It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape (dims,)
        upper: The maximum value of the tensor. Shape (dims,)

    Returns:
        Normalized transform of the tensor. Shape (N, dims)
    """
    # default value of center
    offset = (lower + upper) * 0.5
    # return normalized tensor
    return 2 * (x - offset) / (upper - lower)

def unscale_transform(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Denormalizes a given input tensor from range of [-1, 1] to (lower, upper).

    @note It uses pytorch broadcasting functionality to deal with batched input.

    Args:
        x: Input tensor of shape (N, dims).
        lower: The minimum value of the tensor. Shape (dims,)
        upper: The maximum value of the tensor. Shape (dims,)

    Returns:
        Denormalized transform of the tensor. Shape (N, dims)
    """
    # default value of center
    offset = (lower + upper) * 0.5
    # return normalized tensor
    return x * (upper - lower) * 0.5 + offset

@torch.jit.script
def quat_conjugate(a: torch.Tensor) -> torch.Tensor:
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)

@torch.jit.script
def quat_mul(a, b):
    a, b = torch.broadcast_tensors(a, b)
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat

@torch.jit.script
def quat_mul_point_tensor(quat, pt_input):
    quat_con = quat_conjugate(quat)
    pt_new = quat_mul(quat_mul(quat, pt_input), quat_con)
    return pt_new[:,:3]

@torch.jit.script
def transform_world2target(quat, pt_input):
    # padding point to 4 dim
    B = pt_input.size(0)
    padding = torch.zeros([B,1]).to(pt_input.device)
    pt_input = torch.cat([pt_input,padding],1)

    quat_inverse = quat_conjugate(quat)
    pt_new = -quat_mul_point_tensor(quat_inverse, pt_input)
    return quat_inverse , pt_new

@torch.jit.script
def transform_points(quat, pt_input):
    quat_con = quat_conjugate(quat)
    pt_new = quat_mul(quat_mul(quat, pt_input), quat_con)
    if len(pt_new.size()) == 3:
        return pt_new[:,:,:3]
    elif len(pt_new.size()) == 2:
        return pt_new[:,:3]
    
def transform_target2source(s_quat, s_pos, t_quat, t_pos):
    w2s_quat, w2s_pos = transform_world2target(s_quat, s_pos)
    t2s_pos, t2s_quat = multiply_transform(w2s_pos, w2s_quat, t_pos, t_quat)
    return t2s_pos, t2s_quat

def multiply_transform(s_pos, s_quat, t_pos, t_quat):
    t2s_quat = quat_mul(s_quat, t_quat)

    B = t_pos.size()[0]
    padding = torch.zeros([B, 1]).to(t_pos.device)
    t_pos_pad = torch.cat([t_pos,padding],-1)
    s_quat = s_quat.expand_as(t_quat)
    t2s_pos = transform_points(s_quat, t_pos_pad)
    s_pos = s_pos.expand_as(t2s_pos)
    t2s_pos += s_pos
    return t2s_pos, t2s_quat

@torch.jit.script
def quat_diff_rad(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Get the difference in radians between two quaternions.

    Args:
        a: first quaternion, shape (N, 4)
        b: second quaternion, shape (N, 4)
    Returns:
        Difference in radians, shape (N,)
    """
    b_conj = quat_conjugate(b)
    mul = quat_mul(a, b_conj)
    # 2 * torch.acos(torch.abs(mul[:, -1]))
    return 2.0 * torch.asin(torch.clamp(torch.norm(mul[:, 0:3], p=2, dim=-1), max=1.0))

@torch.jit.script
def get_euler_xyz(q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2 * np.pi), pitch % (2 * np.pi), yaw % (2 * np.pi)

def numpy_scale_transform(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Normalizes a given input array to a range of [-1, 1].

    Args:
        x: Input array of shape (N, dims).
        lower: The minimum value of the array. Shape (dims,)
        upper: The maximum value of the array. Shape (dims,)

    Returns:
        Normalized transform of the array. Shape (N, dims)
    """
    # default value of center
    offset = (lower + upper) * 0.5
    # return normalized array
    return 2 * (x - offset) / (upper - lower)

def numpy_unscale_transform(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Denormalizes a given input array from range of [-1, 1] to (lower, upper).

    Args:
        x: Input array of shape (N, dims).
        lower: The minimum value of the array. Shape (dims,)
        upper: The maximum value of the array. Shape (dims,)

    Returns:
        Denormalized transform of the array. Shape (N, dims)
    """
    # default value of center
    offset = (lower + upper) * 0.5
    # return denormalized array
    return x * (upper - lower) * 0.5 + offset