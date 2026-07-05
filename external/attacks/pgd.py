from typing import Callable
from .attack import Attack
from typing import Any, List

import torch


class PGD(Attack):

    def __init__(self,
                 eps: float = 0.3,
                 alpha: float = 0.1,
                 n_steps: int = 10,
                 random_start: bool = False,
                 targeted: bool = False,
                 loss_fn: "torch.nn.Module" = ...,
                 target_transform_fn: Callable = None,
                 min: List[float] = [0.0, 0.0, 0.0],
                 max: List[float] = [1.0, 1.0, 1.0],
                 attack_ratio: float = 1.0
    ) -> None:
        super().__init__(loss_fn, target_transform_fn, min, max, attack_ratio)
        self.eps = eps
        self.alpha = alpha
        self.n_steps = n_steps
        self.random_start = random_start
        self.targeted = targeted

        self.min_clamp = torch.tensor(min).view(1, 3, 1, 1)
        self.max_clamp = torch.tensor(max).view(1, 3, 1, 1)



    def _compute_adv_samples(self,
                             model: torch.nn.Module,
                             inputs: torch.Tensor,
                             targets: torch.Tensor
    ) -> torch.Tensor:

        adv_inputs = inputs.clone().detach()

         # move min/max clamp to device
        self.min_clamp = self.min_clamp.to(inputs.device)
        self.max_clamp = self.max_clamp.to(inputs.device)

        if self.random_start:
            # Starting at a uniformly random point
            adv_inputs = adv_inputs + torch.empty_like(adv_inputs) \
                                           .uniform_(-self.eps, self.eps)
            adv_inputs = torch.clamp(adv_inputs, min=0, max=1).detach()

        for _ in range(self.n_steps):
            adv_inputs.requires_grad = True
            outputs = model(adv_inputs)

            # Get loss and cost
            cost = None
            if self.targeted:
                output_targets = self.target_transform(outputs)
                cost = -self.loss_fn(targets, output_targets)
            else:
                cost = self.loss_fn(outputs, targets)

            # Update adversarial images
            grad = torch.autograd.grad(cost,
                                       adv_inputs,
                                       retain_graph=False,
                                       create_graph=False)[0]

            adv_inputs = adv_inputs.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_inputs-inputs, min=-self.eps, max=self.eps)
            # adv_inputs = torch.clamp(inputs + delta, min=0, max=1).detach()
            adv_inputs = torch.clamp(inputs + delta, min=self.min_clamp, max=self.max_clamp).detach() # channel-wise clamp

        return adv_inputs