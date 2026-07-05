from typing import Callable
from .attack import Attack
from typing import Any, List

import torch
from training.losses import compute_hint_loss, compute_functional_loss


class IrisPGD(Attack):

    def __init__(self,
                 eps: float = 0.3,
                 alpha: float = 0.1,
                 n_steps: int = 10,
                 random_start: bool = False,
                 loss_fn: "torch.nn.Module" = ...,
                 target_transform_fn: Callable = None,
                 min: List[float] = [0.0, 0.0, 0.0],
                 max: List[float] = [1.0, 1.0, 1.0],
                 attack_ratio: float = 1.0,
                 stitch_layer_index: int = 1,) -> None:
                 
        super().__init__(loss_fn, target_transform_fn, min, max, attack_ratio)
        self.eps = eps
        self.alpha = alpha
        self.n_steps = n_steps
        self.random_start = random_start
        self.stitch_layer_index = stitch_layer_index

        self.min_clamp = torch.tensor(min).view(1, 3, 1, 1)
        self.max_clamp = torch.tensor(max).view(1, 3, 1, 1)


    def generate(self,
                 model: torch.nn.Module, 
                 inputs: torch.Tensor, 
                 verbose: bool = False,
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
        
        # move min/max clamp to device
        self.min_clamp = self.min_clamp.to(device)
        self.max_clamp = self.max_clamp.to(device)

        # Put training model to eval mode
        if is_train:
            model.eval()

        # Generate adversarial samples 
        adv_samples = self._compute_adv_samples(model, 
                                                perturb_inputs,
                                                verbose=verbose)

        # Rewrite perturbed inputs
        final_samples = inputs.clone().detach().to(device)
        final_samples[perturb_idx] = adv_samples

        # Put training model back to train mode
        if is_train:
            model.train()

        return final_samples


    def _compute_adv_samples(self,
                             model: torch.nn.Module,
                             inputs: torch.Tensor,
                             loss_type: str = "hint",
                             verbose = False,
                             ) -> torch.Tensor:




        adv_inputs = inputs.clone().detach()

        if self.random_start:
            
            adv_inputs = torch.empty_like(adv_inputs).uniform_(0, 1)
            adv_inputs = adv_inputs * (self.max_clamp - self.min_clamp) + self.min_clamp 
            

        for step_index in range(self.n_steps):
            adv_inputs.requires_grad = True

            iris_logits = model(adv_inputs)
            iris_feats = model.transform_input

            iris_hooked_activations = {}
            for key, value in model.front_model.hooked_activations.items():
                iris_hooked_activations[key] = value.detach()

            input_logits = model(inputs)
            input_feats = model.transform_input

            input_hooked_activations = {}
            for key, value in model.front_model.hooked_activations.items():
                input_hooked_activations[key] = value.detach()


            _, functional_loss = compute_functional_loss(iris_hooked_activations, 
                                                         input_hooked_activations, 
                                                         forced_output=iris_feats,
                                                         last_m2_out=input_feats,
                                                         stitch_logit=iris_logits,
                                                         end_logit=input_logits,
                                                         stitch_layer_index=self.stitch_layer_index,
                                                         funct_cutoff=None,
                                                         fula_soft=True) 


            hint_loss = compute_hint_loss(iris_feats, input_feats)

            if loss_type == "functional":
                cost = - functional_loss
            elif loss_type == "hint":
                cost = - hint_loss

            if verbose:
                print("Cost: ", - cost.item(), end='\r')


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