# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# adapted from https://github.com/pytorch/opacus/blob/main/opacus/optimizers/optimizer.py

from __future__ import annotations
from typing import Callable, List, Optional, Union
from opacus.optimizers import DPOptimizer
from opacus.optimizers.optimizer import _check_processed_flag, _mark_as_processed
from opt_einsum.contract import contract

import torch
from torch.optim import Optimizer
from torch.distributions.laplace import Laplace
from transformers.utils import logging

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)

class LaplaceDPOptimizer(DPOptimizer):
    def add_noise(self):
        laplace = Laplace(loc=0, scale=self.noise_multiplier * self.max_grad_norm)
        for p in self.params:
            _check_processed_flag(p.summed_grad)

            noise = laplace.sample(p.summed_grad.shape)
            p.grad = p.summed_grad + noise

            _mark_as_processed(p.summed_grad)

class GDPOptimizer(DPOptimizer):
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
        preclip = 0,
        neg_k = 0,
        dp_type = None,
        debug = False,
    ):
        """

        Args:
            optimizer: wrapped optimizer.
            noise_multiplier: noise multiplier
            max_grad_norm: max grad norm used for gradient clipping
            expected_batch_size: batch_size used for averaging gradients. When using
                Poisson sampling averaging denominator can't be inferred from the
                actual batch size. Required is ``loss_reduction="mean"``, ignored if
                ``loss_reduction="sum"``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            generator: torch.Generator() object used as a source of randomness for
                the noise
            secure_mode: if ``True`` uses noise generation approach robust to floating
                point arithmetic attacks.
                See :meth:`~opacus.optimizers.optimizer._generate_noise` for details
        """
        super().__init__(optimizer, 
                         noise_multiplier=noise_multiplier,
                         max_grad_norm=max_grad_norm,
                         expected_batch_size=expected_batch_size,
                         loss_reduction=loss_reduction,
                         generator=generator,
                         secure_mode=secure_mode
        )
        self.preclip = preclip
        self.neg_k = neg_k
        self.dp_type = dp_type
        self.default_max_grad_norm = max_grad_norm
        self.debug = debug
        
        logger.info(f'  Using Relation-DP Optimizer with preclip noise, sigma = {preclip} and {neg_k} negative samples.')

    def _set_per_sample_clip_factor(self, ratio):
        if self.debug:
            print(f'set_per_sample_clip_factor: {ratio}')
        self.max_grad_norm = ratio*self.default_max_grad_norm

    def _reset_clip_factor(self):
        self.max_grad_norm = self.default_max_grad_norm

    def pre_step(
        self, closure: Optional[Callable[[], float]] = None
    ) -> Optional[float]:
        """
        Perform actions specific to ``DPOptimizer`` before calling
        underlying  ``optimizer.step()``

        Args:
            closure: A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.
        """

        # The corner case when the optimizer has no trainable parameters.
        # Essentially the DPOptimizer act as a normal optimizer
        if self.grad_samples is None or len(self.grad_samples) == 0:
            return True

        if self.preclip > 0:
            for p in self.params:
                reference = p.grad_sample
                noise = torch.normal(mean=0, std=self.preclip, size=reference.shape, device=reference.device, generator=None,)
                p.grad_sample += noise
        
        self.clip_and_accumulate()
        if self._check_skip_next_step():
            self._is_last_step_skipped = True
            return False

        self.add_noise()
        self.scale_grad()

        if self.step_hook:
            self.step_hook(self)

        self._is_last_step_skipped = False
        return True
    
    def clip_and_accumulate(self):
        """
        Performs gradient clipping.
        Stores clipped and aggregated gradients into `p.summed_grad```
        """
        if len(self.grad_samples[0]) == 0:
            # Empty batch
            per_sample_clip_factor = torch.zeros(
                (0,), device=self.grad_samples[0].device
            )
        else:
            per_param_norms = [
                g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
            ]
            # clipping for non-relational data
            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
            per_sample_clip_factor = (
                self.max_grad_norm / (per_sample_norms + 1e-6)
            ).clamp(max=1.0)
            
        for p in self.params:
            _check_processed_flag(p.grad_sample)
            # gradient reduction mode -> mean
            grad_sample = self._get_flat_grad_sample(p)
            grad = contract("i,i...", per_sample_clip_factor, grad_sample)

            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        if closure is not None:
            with torch.enable_grad():
                closure()

        if self.pre_step():
            return self.original_optimizer.step()
        else:
            return 'skipped'