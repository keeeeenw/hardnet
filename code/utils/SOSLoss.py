import Losses

def loss_SOSNet(anchor, positive, eps=1e-8, margin=1.0, batch_reduce='min', loss_type="triplet_margin"):
    """calculates postive and minimal negative distance from distance matrix"""
    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Input must be a 2D matrix."
    eps = 1e-8
    ap_dist_matrix = distance_matrix_vector(anchor, positive) + eps
    aa_dist_matrix = distance_matrix_vector(anchor, anchor) + eps
    pp_dist_matrix = distance_matrix_vector(positive, positive) + eps

    # steps to filter out same patches that occur in distance matrix as negatives
    pos1 = torch.diag(ap_dist_matrix)
    ap_dist_matrix_masked = mask_dist_matrix(ap_dist_matrix)
    aa_dist_matrix_masked = mask_dist_matrix(aa_dist_matrix)
    pp_dist_matrix_masked = mask_dist_matrix(pp_dist_matrix)
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
        print ('Unknown batch reduce mode. Try min for SOS loss')
        sys.exit(1)
    return eval_QHTloss(pos, min_neg, eps, margin, batch_reduce, loss_type)


def eval_QHTloss(pos, min_neg, eps, margin, batch_reduce, loss_type):
    if loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
    elif loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos);
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps;
        loss = - torch.log( exp_pos / exp_den )
    elif loss_type == 'contrastive':
        loss = torch.clamp(margin - min_neg, min=0.0) + pos;
    else:
        print ('Unknown loss type. Try triplet_margin, softmax or contrastive.')
        sys.exit(1)
    loss = torch.mean(loss)
    return loss


def mask_dist_matrix(dist_matrix, zero_val=0.008, mask_val=10, no_cuda=False):
    """adds a mask value to diagonal entries and negative entries to dist_matrix"""
    if no_cuda:
        eye = torch.autograd.Variable(torch.eye(ap_dist_matrix.size(1)))
    else:
        eye = torch.autograd.Variable(torch.eye(ap_dist_matrix.size(1))).cuda()
    dist_matrix_masked = dist_matrix + eye * mask_val
    mask = (dist_matrix_masked.ge(zero_val).float() - 1.0) * (-1)
    mask = mask.type_as(dist_matrix_masked) * mask_val
    dist_matrix_masked = dist_matrix_masked + mask
    return dist_matrix_masked
