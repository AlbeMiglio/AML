import numpy as np

# eggbox and glue look identical under some rotations, so a correct pose can put
# point i of the prediction on top of point j of the ground truth. Comparing
# corresponding points would score that as a large error, hence ADD-S for these two.
SYMMETRIC_OBJ_IDS = {10, 11}  # eggbox, glue


def add_distance(pts, R_gt, T_gt, R_pred, T_pred):
    """Mean distance between corresponding transformed model points."""
    # pts holds one point per ROW, so the textbook "R p" is written "p R^T" here.
    p_gt   = pts @ R_gt.T   + T_gt
    p_pred = pts @ R_pred.T + T_pred
    return float(np.mean(np.linalg.norm(p_gt - p_pred, axis=1)))


def adds_distance(pts, R_gt, T_gt, R_pred, T_pred):
    """ADD-S: mean nearest-neighbour distance (for symmetric objects)."""
    p_gt   = pts @ R_gt.T   + T_gt    # (N, 3)
    p_pred = pts @ R_pred.T + T_pred  # (N, 3)
    # Brute-force nearest neighbour: the full 500x500 distance matrix. Cheap enough
    # at this size, and exact -- no KD-tree approximation to justify at review time.
    diff   = p_pred[:, None, :] - p_gt[None, :, :]  # (N, N, 3)
    return float(np.mean(np.linalg.norm(diff, axis=2).min(axis=1)))


def pose_error(pts, R_gt, T_gt, R_pred, T_pred, obj_id):
    """Dispatch ADD-S for symmetric objects, ADD otherwise."""
    fn = adds_distance if obj_id in SYMMETRIC_OBJ_IDS else add_distance
    return fn(pts, R_gt, T_gt, R_pred, T_pred)
