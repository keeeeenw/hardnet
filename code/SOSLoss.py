import torch
import sys
from Losses import distance_matrix_vector


def loss_SOSNet(anchor, positive, eps=1e-8, margin=1.0, batch_reduce='min', no_cuda=False):
    """calculates postive and minimal negative distance from distance matrix"""
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Input must be a 2D matrix."
    ap_dist_matrix = distance_matrix_vector(anchor, positive) + eps
    aa_dist_matrix = distance_matrix_vector(anchor, anchor) + eps
    pp_dist_matrix = distance_matrix_vector(positive, positive) + eps

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(ap_dist_matrix)
    ap_dist_matrix_masked = mask_dist_matrix(ap_dist_matrix, no_cuda=no_cuda)
    aa_dist_matrix_masked = mask_dist_matrix(aa_dist_matrix, no_cuda=no_cuda)
    pp_dist_matrix_masked = mask_dist_matrix(pp_dist_matrix, no_cuda=no_cuda)
    if batch_reduce == 'min':   # 'min' mode is adapted to SOS loss
        min_neg = torch.min(ap_dist_matrix_masked,1)[0]
        min_neg2 = torch.min(ap_dist_matrix_masked,0)[0]
        min_neg3 = torch.min(aa_dist_matrix_masked,1)[0]
        min_neg4 = torch.min(pp_dist_matrix_masked,1)[0]
        min_neg = torch.min(min_neg,min_neg2)
        min_neg = torch.min(min_neg,min_neg3)
        min_neg = torch.min(min_neg,min_neg4)
        pos = pos1
    else:
        print('Unknown batch reduce mode. Try min for SOS loss')
        sys.exit(1)

    # regularization term
    _, a_inds = aa_dist_matrix_masked.topk(8)    # tensor of size [N, 8]
    _, p_inds = pp_dist_matrix_masked.topk(8)    # tensor of size [N, 8]
    # take per-row union of a_inds and p_inds
    union_inds = torch.cat((a_inds, p_inds), dim=1)
    r = 0
    for i in range(union_inds.shape[0]):
        union_inds_i = torch.unique(union_inds[i])  # length of union_inds_i varies
        r_i = 0
        for j in union_inds_i:
            r_i += torch.pow(aa_dist_matrix[i, j] - pp_dist_matrix[i, j], 2)  # ( d(ai, aj) - d(pi, pj) )^2
        r += torch.sqrt(r_i)
    r = r / anchor.shape[0]
    # end of regularization term

    loss = torch.pow(torch.clamp(margin + pos - min_neg, min=0.0), 2)
    return torch.mean(loss) + r


def mask_dist_matrix(dist_matrix, zero_val=0.008, mask_val=10, no_cuda=False):
    """adds a mask value to diagonal entries and negative entries to dist_matrix"""
    if no_cuda:
        eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1)))
        # eye = torch.eye(dist_matrix.size(1))
    else:
        eye = torch.autograd.Variable(torch.eye(dist_matrix.size(1))).cuda()
        # eye = torch.eye(dist_matrix.size(1)).cuda()
    dist_matrix_masked = dist_matrix + eye * mask_val
    mask = (dist_matrix_masked.ge(zero_val).float() - 1.0) * (-1)
    mask = mask.type_as(dist_matrix_masked) * mask_val
    dist_matrix_masked = dist_matrix_masked + mask
    return dist_matrix_masked
