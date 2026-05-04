import torch
import torch.nn as nn

class ADDLoss(nn.Module):
    def __init__(self):
        super(ADDLoss, self).__init__()

    def forward(self, pred_R, pred_T, gt_R, gt_T, model_points):
        """
        pred_R: (B, 9) - Flattened 3x3 rotation matrix
        pred_T: (B, 3) - Predicted 3D translation
        gt_R: (B, 3, 3) - Ground truth rotation matrix
        gt_T: (B, 3) - Ground truth translation
        model_points: (B, N, 3) - 3D CAD model points
        """
        pred_R = pred_R.view(-1, 3, 3)
        
        # Transform points with predicted pose: R*p + T
        pred_points = torch.bmm(model_points, pred_R.transpose(1, 2)) + pred_T.unsqueeze(1)
        
        # Transform points with GT pose: R_gt*p + T_gt
        gt_points = torch.bmm(model_points, gt_R.transpose(1, 2)) + gt_T.unsqueeze(1)

        # ADD metric: mean distance between corresponding points
        loss = torch.mean(torch.norm(pred_points - gt_points, dim=2))
            
        return loss