import numpy as np

import torch
import torch.nn.functional as F

import scipy.stats

from utils import unwrap

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    output = torch.as_tensor(output)
    target = torch.as_tensor(target)

    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]


def soft_accuracy(output, target):

    output = torch.as_tensor(output)
    target = torch.as_tensor(target)

    batch_size = target.size(0)

    output = F.softmax(output, dim=1)

    p_correct = output.gather(1, target.unsqueeze(1)).sum()

    return p_correct * 100.0 / batch_size

def agreement(output_a, output_b, target, soft_normalization=True):

    if output_a.shape != output_b.shape:
        
        hard_agreement = soft_agreement = None
        
        return hard_agreement, soft_agreement

    output_a = torch.as_tensor(output_a)
    output_b = torch.as_tensor(output_b)
    target = torch.as_tensor(target)

    output_a, output_b = F.softmax(output_a, dim=1), F.softmax(output_b, dim=1)

    maxk = max((1,))
    batch_size = target.size(0)

    _, pred_a = output_a.topk(maxk, 1, True, True)
    pred_a = pred_a.t()

    _, pred_b = output_b.topk(maxk, 1, True, True)
    pred_b = pred_b.t()

    correct_a = pred_a.eq(target.reshape(1, -1).expand_as(pred_a))
    correct_b = pred_b.eq(target.reshape(1, -1).expand_as(pred_b))

    # c_obs (hard)
    hard_agreement = (correct_a == correct_b).sum()

    # c_obs (soft)
    soft_agreement = (output_a * output_b).sum(dim=1)

    if soft_normalization:
        soft_agreement_optimal = torch.max((output_b * output_b).sum(dim=1), (output_a * output_a).sum(dim=1))
        soft_agreement = (soft_agreement / soft_agreement_optimal)

    soft_agreement = soft_agreement.sum()

    return hard_agreement * 100.0 / batch_size , soft_agreement * 100.0 / batch_size

def hard_error_consistency(agreement_ratio, acc_a, acc_b):


    c_exp = acc_a * acc_b + (1 - acc_a) * (1 - acc_b)
    c_obs = agreement_ratio

    if c_exp == 1.0:   
        return 1.0
    
    return (c_obs - c_exp) / (1 - c_exp)

def soft_error_consistency(agreement_prob, soft_a, soft_b, num_classes=10):


    c_exp = soft_a * soft_b + (1 - soft_a) * (1 - soft_b) * (1 / (num_classes - 1)) 
    c_obs = agreement_prob

    if c_exp == 1.0:
        return 1.0

    return (c_obs - c_exp) / (1 - c_exp)


def _get_matrix_numerical_rank(matrix, sum_thold=0.95, max_sigma_thold=None):

    S  = np.linalg.svd(matrix, compute_uv=False, hermitian=False)
    condition_number = S[0] / S[-1]


    if max_sigma_thold:
        sigma_max = S[0]
        sigma_threshold = sigma_max * max_sigma_thold 
        numerical_rank = np.sum(S > sigma_threshold)

    else:
        ## numerical rank as the number of singular values that comprise 95% of the nuclear norm
        nuclear_norm = S.sum()
        cumulative_sum = np.cumsum(S, axis=0)
        threshold = sum_thold * nuclear_norm

        numerical_rank = np.sum(cumulative_sum <= threshold)


    return condition_number, numerical_rank


def compute_rank_affine(transform_weight, sum_thold=0.95, max_sigma_thold=None):

    condition_number = torch.linalg.cond(transform_weight).item()
    full_rank = transform_weight.shape[0]

    condition_number, numerical_rank = _get_matrix_numerical_rank(matrix=transform_weight.data.cpu().numpy(), 
                                                                 sum_thold=sum_thold,
                                                                 max_sigma_thold=max_sigma_thold)
    
    relative_rank = (numerical_rank / full_rank)

    affine_rank_metrics = {'numerical_rank': float(numerical_rank),
                           'condition_number': float(condition_number),
                           'full_rank': float(full_rank),
                           'relative_rank': float(relative_rank)}

    return affine_rank_metrics

def compute_rank_representation(feats, num_samples=10_000, num_feats=8_000, sum_thold=0.95, max_sigma_thold=None):


    ## Randomly permute rows
    if feats.size(0) > num_samples:
        perm = torch.randperm(feats.size(0))
        feats = feats[perm, :]
        feats = feats[:num_samples, :]


    # Randomly permute columns
    perm = torch.randperm(feats.size(1))
    feats = feats[:, perm]
    feats = feats[:, :num_feats]
    
    ## centralize feats
    feats = feats - feats.mean(dim=0)

    ## svd on the covariance
    cov = torch.cov(feats.T)

    _, numerical_rank  = _get_matrix_numerical_rank(matrix=cov.data.cpu().numpy(), 
                                                    sum_thold=sum_thold,
                                                    max_sigma_thold=max_sigma_thold)

    return numerical_rank / num_feats

def compute_rank_data(dataloader, model, num_samples=10_000, num_feats=8_000, sum_thold=0.95, max_sigma_thold=None, model_instance="stitch"):


    feats = []

    for image, _ in dataloader:

        image = image.cuda(non_blocking=True)

        if model_instance=="stitch":
            _ = model(image)
            _feats = unwrap(model).forced_output.view(len(image), -1).cpu()
        elif model_instance=="end":
            unwrap(model).reset_stitch_connection()
            unwrap(model).end_model(image)
            _feats = unwrap(model).last_m2_out.view(len(image), -1).cpu()
        else:
            raise ValueError("model_instance should be either 'stitch' or 'end' ")

        feats.append(_feats)

    feats = torch.cat(feats, dim=0)
    data_rank = compute_rank_representation(feats, num_samples=num_samples, num_feats=num_feats, 
                                            sum_thold=sum_thold, max_sigma_thold=max_sigma_thold)

    return float(data_rank)


def get_grad_statistics(grads):

    statistics = {
        'mean': np.mean(grads),
        'variance': np.var(grads),
        'min': np.min(grads),
        'max': np.max(grads),
        'range': np.ptp(grads),  # range (max - min)
        'skewness': scipy.stats.skew(grads),
        'kurtosis': scipy.stats.kurtosis(grads),
    }
    return statistics
