from scipy.spatial.distance import pdist
import numpy as np
import torch


def compute_all_metrics(pred, gt, gt_multi):
    """
    calculate all metrics

    Args:
        pred: candidate prediction, shape as [50, t_pred, 3 * joints_num]
        gt: ground truth, shape as [1, t_pred, 3 * joints_num]
        gt_multi: multi-modal ground truth, shape as [multi_modal, t_pred, 3 * joints_num]

    Returns:
        diversity, ade, fde, mmade, mmfde, ade_m,fde_m, mmade_m, mmfde_m, ade_w, fde_w, mmade_w, mmfde_w
    """
    if pred.shape[0] == 1:
        diversity = 0.0
    dist_diverse = torch.pdist(pred.reshape(pred.shape[0], -1))
    diversity = dist_diverse.mean()

    gt_multi = torch.from_numpy(gt_multi).to('cuda')
    gt_multi_gt = torch.cat([gt_multi, gt], dim=0)

    gt_multi_gt = gt_multi_gt[None, ...]
    pred = pred[:, None, ...]

    diff_multi = pred - gt_multi_gt
    dist = torch.linalg.norm(diff_multi, dim=3)

    mmfde, _ = dist[:, :-1, -1].min(dim=0)
    mmfde = mmfde.mean()
    mmade, _ = dist[:, :-1].mean(dim=2).min(dim=0)
    mmade = mmade.mean()

    ade, _ = dist[:, -1].mean(dim=1).min(dim=0)
    fde, _ = dist[:, -1, -1].min(dim=0)
    ade = ade.mean()
    fde = fde.mean()

    mmfde_w, _ = dist[:, :-1, -1].max(dim=0)
    mmfde_w = mmfde_w.mean()
    mmade_w, _ = dist[:, :-1].mean(dim=2).max(dim=0)
    mmade_w = mmade_w.mean()

    ade_w, _ = dist[:, -1].mean(dim=1).max(dim=0)
    fde_w, _ = dist[:, -1, -1].max(dim=0)
    ade_w = ade_w.mean()
    fde_w = fde_w.mean()

    mmfde_m, _ = dist[:, :-1, -1].median(dim=0)
    mmfde_m = mmfde_m.mean()
    mmade_m, _ = dist[:, :-1].mean(dim=2).median(dim=0)
    mmade_m = mmade_m.mean()

    ade_m, _ = dist[:, -1].mean(dim=1).median(dim=0)
    fde_m, _ = dist[:, -1, -1].median(dim=0)
    ade_m = ade_m.mean()
    fde_m = fde_m.mean()

    return diversity, ade, fde, mmade, mmfde, ade_m,fde_m, mmade_m, mmfde_m, ade_w, fde_w, mmade_w, mmfde_w
