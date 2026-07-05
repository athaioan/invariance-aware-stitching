# trex
# Copyright (C) 2023-present NAVER Corp.
# CC BY-NC-SA 4.0

##################################################
# Utility routines (functions and classes) used for training models.
# Some of these routines are re-used from
# - DINO (https://github.com/facebookresearch/dino)
# - MoCo (https://github.com/facebookresearch/moco)
# - PyTorch examples (https://github.com/pytorch/examples)
##################################################


import os
import random

import yaml

import numpy as np
import torch
import torch.nn as nn

def fix_random_seeds(seed=22):
    """
    Fix random seeds.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print(
                    "=> loaded '{}' from checkpoint '{}' with msg {}".format(
                        key, ckp_path, msg
                    )
                )
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print(
                        "=> failed to load '{}' from checkpoint: '{}'".format(
                            key, ckp_path
                        )
                    )
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def write_metrics_to_yaml(metrics, output_dir):

    with open(os.path.join(output_dir, "results.yaml"), "w") as file:
        for key, value in metrics.items():
            temp_dict = {key: value}
            yaml.dump(temp_dict, file, sort_keys=False, default_flow_style=False)
            file.write('\n')
            
    return



def contribution_feature_map(feature):

    pooled_gradients = torch.mean(feature.grad, dim=[2, 3])

    weighted_feature =  feature * pooled_gradients[:, :, None, None]

    heatmap = torch.sum(weighted_feature, dim=1)

    return heatmap


def freeze_bn_running_stats(model):

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.SyncBatchNorm):
            module.momentum = 0
    return 

def unfreeze_bn_running_stats(model, momentum=0.1):

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.SyncBatchNorm):
            module.momentum = momentum
    return