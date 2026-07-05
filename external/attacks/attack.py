from abc import ABC, abstractmethod
from re import S
from typing import Any, Callable, Dict
import numpy as np

import torch

from art.attacks.evasion import  AutoAttack, AutoProjectedGradientDescent
from art.attacks import EvasionAttack
from art.estimators.classification import PyTorchClassifier


class Attack(ABC):
    
    def __init__(self, 
                 loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(),
                 target_transform_fn: Callable = None,
                 min: float = 0.0,
                 max: float = 1.0,
                 attack_ratio: float = .5
    ) -> None:
        super().__init__()
        self.loss_fn = loss_fn
        self.target_transform_fn = target_transform_fn
        self.min = min
        self.max = max
        self.attack_ratio = attack_ratio

    @abstractmethod
    def _compute_adv_samples(self, 
                             model: torch.nn.Module, 
                             inputs: torch.Tensor, 
                             targets: torch.Tensor
    ) -> torch.Tensor:
        # Override this method in subclasses
        raise NotImplementedError()

    def generate(self,
                 model: torch.nn.Module, 
                 inputs: torch.Tensor, 
                 targets: torch.Tensor,
                 *args: Any,
                 **kwargs: Any
    ) -> torch.Tensor:
        # Get model parameters
        device = next(model.parameters()).device
        is_train = model.training

        # Select inputs to perturb
        n_samples = len(inputs)
        n_perturb_samples = int(n_samples * self.attack_ratio)
        perturb_idx = torch.randperm(n_samples)[:n_perturb_samples]

        # Clone batch for safety reasons
        perturb_inputs = inputs[perturb_idx].clone().detach().to(device)
        perturb_targets = targets[perturb_idx].clone().detach().to(device)

        # Put training model to eval mode
        if is_train:
            model.eval()

        # Generate adversarial samples 
        adv_samples = self._compute_adv_samples(model, 
                                                perturb_inputs, 
                                                perturb_targets)

        # Rewrite perturbed inputs
        final_samples = inputs.clone().detach().to(device)
        final_samples[perturb_idx] = adv_samples

        # Put training model back to train mode
        if is_train:
            model.train()

        return final_samples

    def target_transform(self, output: torch.Tensor) -> torch.Tensor:
        if self.target_transform_fn is not None:
            return self.target_transform_fn(output)
        raise Exception("Target transform function not set!")
    

def get_attacks(model: PyTorchClassifier, 
                str_dataset: str = "mnist"
) -> Dict[str, EvasionAttack]:
    # Dataset-independent parameters

    norm = np.inf

    # Dataset-dependent parameters
    # Linf attack eps
    if str_dataset.lower() in ["cifar10", "cifar100", "svhn", "ls_imagenet10", "ls_imagenet100"]:
        max_iter = 20
        n_random_init = 1
        batch_size = 64
        eps = 8/255 
        eps_step = 2/255

    elif str_dataset.lower() in ["imagenet100", "imagenetB100"]:

        max_iter = 20 
        n_random_init = 1
        batch_size = 16
        eps =  4 * 4.758/255 # 4/255
        eps_step = 1 * 4.758/255 # 1/255
   


    autopgd_ce = AutoProjectedGradientDescent(estimator=model,
                                              norm=norm,
                                              eps=eps,
                                              eps_step=eps_step,
                                              max_iter=max_iter,
                                              targeted=False,
                                              nb_random_init=n_random_init,
                                              batch_size=batch_size,
                                              loss_type="cross_entropy",
                                              verbose=False)

    autopgd_logits = AutoProjectedGradientDescent(estimator=model,
                                                  norm=norm,
                                                  eps=eps,
                                                  eps_step=eps_step,
                                                  max_iter=max_iter,
                                                  targeted=False,
                                                  nb_random_init=n_random_init,
                                                  batch_size=batch_size,
                                                  loss_type="difference_logits_ratio",
                                                  verbose=False)


    auto_attack = AutoAttack(estimator=model,
                             eps=eps,
                             eps_step=eps_step,
                             batch_size=batch_size,
                             norm=norm,
                             attacks=[autopgd_ce, autopgd_logits])

    attacks = {'autoattack': auto_attack}

    return attacks