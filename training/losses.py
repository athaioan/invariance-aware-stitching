import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_soft_target_loss(logit, end_logit, epsilon=1e-8):

    soft_targets  = nn.functional.softmax(end_logit, dim=-1)
    soft_prob = nn.functional.log_softmax(logit, dim=-1)

    # soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0]
    soft_targets_loss = torch.sum(soft_targets * ((soft_targets+epsilon).log() - soft_prob)) / soft_prob.size()[0]

    return soft_targets_loss

def compute_hint_loss(forced_output, last_m2_out):
  
    match_diff = forced_output - last_m2_out

    hint_loss = (torch.norm(match_diff, "fro")) / (torch.norm(last_m2_out, "fro")) 

    return hint_loss


def compute_functional_loss(stitch_activations, end_activations, 
                            forced_output, last_m2_out, 
                            stitch_logit, end_logit,
                            stitch_layer_index, funct_cutoff=None,
                            fula_soft=False,
                            plot=None):

    def _dist(x, y):
        return torch.norm(x - y, "fro") / torch.norm(y, "fro")
    
    def _r2_score(x, y, eps=1e-6):
        
        if len(x.shape) == 4:
            ## feat_map
            x = x.permute(0, 2, 3, 1) ## BxCxHxW -> BxHxWxC
            y = y.permute(0, 2, 3, 1) ## BxCxHxW -> BxHxWxC
        else:
            ## feat_vect
            # either BxC or BxTxC
            pass

        x = x.reshape(-1, x.shape[-1])
        y = y.reshape(-1, y.shape[-1])

        ss_res = torch.mean((y - x) ** 2)
        ss_tot = torch.mean((y - y.mean()) ** 2)

        r2 = 1 - (ss_res / (ss_tot + eps))

        return torch.mean(r2)



    functional_loss = []
    explained_variance = []
    for index, (stitch_activation, end_activation) in enumerate(zip(stitch_activations.values(), end_activations.values())):
        
        if (index+1) == stitch_layer_index:
            functional_loss.append(_dist(forced_output, last_m2_out))
            explained_variance.append(_r2_score(forced_output, last_m2_out))
        elif (index+1) > stitch_layer_index:

            if funct_cutoff is not None and (index + 1 - stitch_layer_index) > funct_cutoff:     
                pass
            else:          
                functional_loss.append(_dist(stitch_activation, end_activation))
                explained_variance.append(_r2_score(stitch_activation, end_activation))


    if fula_soft:
        functional_loss.append(compute_soft_target_loss(stitch_logit, end_logit))

    if plot is not None:
        plt.figure()
        plt.scatter(range(len(functional_loss)), [fl.item() for fl in functional_loss])
        # plt.savefig(f'{plot}/functional_loss.png')
        plt.savefig('./functional_loss.png')
        plt.close('all')
# 
        plt.figure()
        plt.scatter(range(len(explained_variance)), [ev.item() for ev in explained_variance])
        # plt.savefig(f'{plot}/explained_variance.png')
        plt.savefig('./explained_variance.png')
        plt.close('all')

    return torch.stack(functional_loss), torch.stack(functional_loss).mean()


# taken from https://github.com/dkobak/simclr-cifar10/
def infoNCE_loss(features, temperature=0.5):
    x = F.normalize(features)
    cos_xx = x @ x.T / temperature
    cos_xx.fill_diagonal_(float("-inf"))
    
    batch_size = cos_xx.size(0) // 2
    targets = torch.arange(batch_size * 2, dtype=int, device=cos_xx.device)
    targets[:batch_size] += batch_size
    targets[batch_size:] -= batch_size

    return F.cross_entropy(cos_xx, targets)