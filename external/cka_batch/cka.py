# marginally modified from https://github.com/numpee/CKA.pytorch

from __future__ import annotations

from typing import Tuple, Optional, Callable, Type, Union, TYPE_CHECKING, List

import torch
import torch.nn as nn
from tqdm.autonotebook import tqdm
from torchmetrics import Metric

from evaluation.metrics import agreement
from utils import MetricLogger



class AccumTensor(Metric):
    def __init__(self, default_value: torch.Tensor):
        super().__init__()

        self.add_state("val", default=default_value, dist_reduce_fx="sum")

    def update(self, input_tensor: torch.Tensor):
        self.val += input_tensor

    def compute(self):
        return self.val

if TYPE_CHECKING:
    from torch.utils.data import DataLoader


class CKA:
    def __init__(self, model: nn.Module, dataloader: DataLoader,
                 num_epochs: int = 10, 
                 group_size: int = 512, epsilon: float = 1e-4, is_main_process: bool = True) -> None:
        """
        Class to extract intermediate features and calculate CKA Matrix.
        :param model: stitch model containing front and end models.
        :param dataloader: Torch DataLoader for dataloading. Assumes first return value contains input images.
        :param num_epochs: Number of epochs for cka_batch. Default: 10
        :param group_size: group_size for GPU acceleration. Default: 512
        :param epsilon: Small multiplicative value for HSIC. Default: 1e-4
        :param is_main_process: is current instance main process. Default: True
        """
        self.model = model
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.group_size = group_size
        self.epsilon = epsilon
        self.is_main_process = is_main_process

        model.eval()


        self.num_layers_X = None
        self.num_layers_Y = None
        self.num_elements = None

        # Metrics to track
        self.cka_matrix = None
        self.hsic_matrix = None
        self.self_hsic_x = None
        self.self_hsic_y = None

    @torch.no_grad()
    def calculate_cka_matrix(self) -> torch.Tensor:

        def _gram(x: torch.Tensor) -> torch.Tensor:
            return x.matmul(x.t())

        curr_hsic_matrix = None
        curr_self_hsic_x = None
        curr_self_hsic_y = None
        for epoch in range(self.num_epochs):
            loader = tqdm(self.dataloader, desc=f"Epoch {epoch}", disable=not self.is_main_process)
            for it, (imgs, *_) in enumerate(loader):
                imgs = imgs.cuda(non_blocking=True)

                self.model.front_model(imgs)
                all_layer_X = self.model.transform_input.flatten(1).detach()
                all_layer_X = [_gram(all_layer_X)] # include to list for compatibility

                # self.forced_output = self.transform_input
                self.model.reset_stitch_connection() 
                # get end representation
                self.model.end_model(imgs)
                all_layer_Y = self.model.last_m2_out.flatten(1).detach()
                all_layer_Y = [_gram(all_layer_Y)] # include to list for compatibility

                # Initialize values on first loop
                if self.num_layers_X is None:
                    curr_hsic_matrix, curr_self_hsic_x, curr_self_hsic_y = self._init_values(all_layer_X, all_layer_Y)

                # Get self HSIC values --> HSIC(K, K), HSIC(L, L)
                self._calculate_self_hsic(all_layer_X, all_layer_Y, curr_self_hsic_x, curr_self_hsic_y)

                # Get cross HSIC values --> HSIC(K, L)
                self._calculate_cross_hsic(all_layer_X, all_layer_Y, curr_hsic_matrix)

                curr_hsic_matrix.fill_(0)
                curr_self_hsic_x.fill_(0)
                curr_self_hsic_y.fill_(0)

        # Update values across GPUs
        hsic_matrix = self.hsic_matrix.compute()
        hsic_x = self.self_hsic_x.compute()
        hsic_y = self.self_hsic_y.compute()
        self.cka_matrix = hsic_matrix.reshape(self.num_layers_Y, self.num_layers_X) / torch.sqrt(hsic_x * hsic_y)
        # print(self.cka_matrix.diagonal())
        # self.cka_matrix = self.cka_matrix.flip(0)
        return self.cka_matrix
    
    def hsic1(self, K: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        '''
        Batched version of HSIC.
        :param K: Size = (B, N, N) where N is the number of examples and B is the group/batch size
        :param L: Size = (B, N, N) where N is the number of examples and B is the group/batch size
        :return: HSIC tensor, Size = (B)
        '''
        assert K.size() == L.size()
        assert K.dim() == 3
        K = K.clone()
        L = L.clone()
        n = K.size(1)

        # K, L --> K~, L~ by setting diagonals to zero
        K.diagonal(dim1=-1, dim2=-2).fill_(0)
        L.diagonal(dim1=-1, dim2=-2).fill_(0)

        KL = torch.bmm(K, L)
        trace_KL = KL.diagonal(dim1=-1, dim2=-2).sum(-1).unsqueeze(-1).unsqueeze(-1)
        middle_term = K.sum((-1, -2), keepdim=True) * L.sum((-1, -2), keepdim=True)
        middle_term /= (n - 1) * (n - 2)
        right_term = KL.sum((-1, -2), keepdim=True)
        right_term *= 2 / (n - 2)
        main_term = trace_KL + middle_term - right_term
        hsic = main_term / (n ** 2 - 3 * n)
        return hsic.squeeze(-1).squeeze(-1)

    def reset(self) -> None:
        # Set values to none, clear features
        self.cka_matrix = None
        self.hsic_matrix = None
        self.self_hsic_x = None
        self.self_hsic_y = None


    def _init_values(self, all_layer_X, all_layer_Y):
        self.num_layers_X = len(all_layer_X)
        self.num_layers_Y = len(all_layer_Y)
        self.num_elements = self.num_layers_Y * self.num_layers_X
        curr_hsic_matrix = torch.zeros(self.num_elements).cuda()
        curr_self_hsic_x = torch.zeros(1, self.num_layers_X).cuda()
        curr_self_hsic_y = torch.zeros(self.num_layers_Y, 1).cuda()
        self.hsic_matrix = AccumTensor(torch.zeros_like(curr_hsic_matrix)).cuda()
        self.self_hsic_x = AccumTensor(torch.zeros_like(curr_self_hsic_x)).cuda()
        self.self_hsic_y = AccumTensor(torch.zeros_like(curr_self_hsic_y)).cuda()
        return curr_hsic_matrix, curr_self_hsic_x, curr_self_hsic_y

    def _calculate_self_hsic(self, all_layer_X, all_layer_Y, curr_self_hsic_x, curr_self_hsic_y):
        for start_idx in range(0, self.num_layers_X, self.group_size):
            end_idx = min(start_idx + self.group_size, self.num_layers_X)
            K = torch.stack([all_layer_X[i] for i in range(start_idx, end_idx)], dim=0)
            curr_self_hsic_x[0, start_idx:end_idx] += self.hsic1(K, K) * self.epsilon
        for start_idx in range(0, self.num_layers_Y, self.group_size):
            end_idx = min(start_idx + self.group_size, self.num_layers_Y)
            L = torch.stack([all_layer_Y[i] for i in range(start_idx, end_idx)], dim=0)
            curr_self_hsic_y[start_idx:end_idx, 0] += self.hsic1(L, L) * self.epsilon

        self.self_hsic_x.update(curr_self_hsic_x)
        self.self_hsic_y.update(curr_self_hsic_y)

    def _calculate_cross_hsic(self, all_layer_X, all_layer_Y, curr_hsic_matrix):
        for start_idx in range(0, self.num_elements, self.group_size):
            end_idx = min(start_idx + self.group_size, self.num_elements)
            K = torch.stack([all_layer_X[i % self.num_layers_X] for i in range(start_idx, end_idx)], dim=0)
            L = torch.stack([all_layer_Y[j // self.num_layers_X] for j in range(start_idx, end_idx)], dim=0)
            curr_hsic_matrix[start_idx:end_idx] += self.hsic1(K, L) * self.epsilon
        self.hsic_matrix.update(curr_hsic_matrix)


def gram(x: torch.Tensor) -> torch.Tensor:
    return x.matmul(x.t())