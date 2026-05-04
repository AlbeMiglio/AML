import numpy as np

SYMMETRIC_OBJ_IDS = {10, 11}  # eggbox, glue


def add_distance(pts, R_gt, T_gt, R_pred, T_pred):
    """Mean distance between corresponding transformed model points."""
    p_gt   = pts @ R_gt.T   + T_gt
    p_pred = pts @ R_pred.T + T_pred
    return float(np.mean(np.linalg.norm(p_gt - p_pred, axis=1)))


def adds_distance(pts, R_gt, T_gt, R_pred, T_pred):
    """ADD-S: mean nearest-neighbour distance (for symmetric objects)."""
    p_gt   = pts @ R_gt.T   + T_gt    # (N, 3)
    p_pred = pts @ R_pred.T + T_pred  # (N, 3)
    diff   = p_pred[:, None, :] - p_gt[None, :, :]  # (N, N, 3)
    return float(np.mean(np.linalg.norm(diff, axis=2).min(axis=1)))


def pose_error(pts, R_gt, T_gt, R_pred, T_pred, obj_id):
    """Dispatch ADD-S for symmetric objects, ADD otherwise."""
    fn = adds_distance if obj_id in SYMMETRIC_OBJ_IDS else add_distance
    return fn(pts, R_gt, T_gt, R_pred, T_pred)
