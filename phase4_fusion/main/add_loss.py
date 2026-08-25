import torch
import torch.nn as nn

class ADDLoss(nn.Module):
    def __init__(self):
        super(ADDLoss, self).__init__()

    def forward(self, pred_R, pred_T, gt_R, gt_T, model_points, per_sample=False):
        """
        pred_R: (B, 9) - Flattened 3x3 rotation matrix
        pred_T: (B, 3) - Predicted 3D translation
        gt_R: (B, 3, 3) - Ground truth rotation matrix
        gt_T: (B, 3) - Ground truth translation
        model_points: (B, N, 3) - 3D CAD model points
        per_sample: return the (B,) per-object ADD instead of the batch mean. Every
            sample carries the same number of points, so the mean of the per-sample
            means is exactly the batch mean — training is unaffected.
        """
        pred_R = pred_R.view(-1, 3, 3)

        # Mathematical ADD: Transform 500 3D CAD points using the network's predicted rotation and translation
        pred_points = torch.bmm(model_points, pred_R.transpose(1, 2)) + pred_T.unsqueeze(1)

        # Ground Truth Transform: Move the exact same points using the real world target pose
        gt_points = torch.bmm(model_points, gt_R.transpose(1, 2)) + gt_T.unsqueeze(1)

        # Objective Function: Calculate the physical distance (L2 norm in meters) between predicted points and real points,
        # perfectly fusing both rotation and translation errors into a single metric without needing arbitrary Lambda weights.
        per_sample_add = torch.norm(pred_points - gt_points, dim=2).mean(dim=1)

        return per_sample_add if per_sample else per_sample_add.mean()