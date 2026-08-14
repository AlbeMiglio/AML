import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R


def matrix_to_quaternion(matrix_3x3):
    """Convert a 3x3 rotation matrix to quaternion (x, y, z, w)."""
    r = R.from_matrix(matrix_3x3)
    return torch.tensor(r.as_quat(), dtype=torch.float32)


def rotation_loss(q_pred, q_true):
    """1 - |q_pred · q_true|, invariant to quaternion sign ambiguity."""
    q_pred = q_pred / torch.norm(q_pred, dim=1, keepdim=True)
    q_true = q_true / torch.norm(q_true, dim=1, keepdim=True)
    inner_prod = torch.abs(torch.sum(q_pred * q_true, dim=1))
    return torch.mean(1 - inner_prod)


def translation_loss(t_pred, t_true):
    """MSE between predicted and ground-truth translation (both in meters)."""
    return F.mse_loss(t_pred, t_true)

def compute_pinhole_translation(bbox, intrinsics, real_diameter):
    """Compute 2D to 3D geometric translation using pinhole camera model."""
    x, y, w, h = bbox
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    # Estimate Z (depth) using real object diameter
    pixel_size = max(w, h)
    Z = (fx * real_diameter) / pixel_size

    # Inverse projection for X and Y
    u_center = x + w / 2
    v_center = y + h / 2
    X = (u_center - cx) * Z / fx
    Y = (v_center - cy) * Z / fy

    return torch.tensor([X, Y, Z])