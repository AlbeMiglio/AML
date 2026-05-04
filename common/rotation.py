import torch
import torch.nn.functional as F


def rot6d_to_matrix(r6d):
    """Convert 6D rotation representation to a valid SO(3) matrix via Gram-Schmidt.

    The 6D vector encodes the first two columns of the target rotation matrix.
    The third column is recovered via cross product, guaranteeing det(R)=+1.

    Args:
        r6d: (B, 6) — two concatenated 3D vectors [a1 | a2]

    Returns:
        R: (B, 3, 3) — valid rotation matrix (columns are orthonormal basis)

    Reference: Zhou et al. "On the Continuity of Rotation Representations in
    Neural Networks", CVPR 2019.
    """
    a1 = r6d[:, :3]  # (B, 3)
    a2 = r6d[:, 3:]  # (B, 3)

    b1 = F.normalize(a1, p=2, dim=1)
    b2 = F.normalize(a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1, p=2, dim=1)
    b3 = torch.cross(b1, b2, dim=1)

    return torch.stack([b1, b2, b3], dim=2)  # columns are b1, b2, b3
