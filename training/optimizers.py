def get_params_groups(model, wd_regularize_all=False):
    """
    Returns two parameters group, one for regularized parameters with weight decay,
    and another for unregularized parameters.
    Unregularized parameters include: Bias terms and batch-norm parameters.
    """
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)

    if wd_regularize_all:
        regularized.extend(not_regularized)
        not_regularized = []

    return [{"params": regularized}, {"params": not_regularized, "weight_decay": 0.0}]
