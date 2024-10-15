from __future__ import annotations

from typing import Tuple, Dict, List

from functools import partial

import torch
import torch.nn as nn
from opacus.grad_sample.functorch import ft_compute_per_sample_gradient, prepare_layer
from opacus.layers.dp_rnn import RNNLinear
from opacus.utils.module_utils import trainable_parameters
from opacus.grad_sample import GradSampleModule, create_or_accumulate_grad_sample
from opacus.grad_sample.grad_sample_module import promote_current_grad_sample, _get_batch_size
from opacus.layers.dp_rnn import DPGRU, DPLSTM, DPRNN, RNNLinear

import importlib.metadata

class GDPGradSampleModule(GradSampleModule):
    def __init__(
        self,
        m: nn.Module,
        *,
        batch_first=True,
        loss_reduction="mean",
        strict: bool = True,
        force_functorch=False,
        group_size=-1,
    ):
        """

        Args:
            m: nn.Module to be wrapped
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            strict: If set to ``True``, the input module will be validated to check that
                ``GradSampleModule`` has grad sampler functions for all submodules of
                the input module (i.e. if it knows how to calculate per sample gradients)
                for all model parameters. If set to ``False``, per sample gradients will
                be computed on "best effort" basis - they will be available where
                possible and set to None otherwise. This is not recommended, because
                some unsupported modules (e.g. BatchNorm) affect other parameters and
                invalidate the concept of per sample gradients for the entire model.
            force_functorch: If set to ``True``, will use functorch to compute
                all per sample gradients. Otherwise, functorch will be used only
                for layers without registered grad sampler methods.

        Raises:
            NotImplementedError
                If ``strict`` is set to ``True`` and module ``m`` (or any of its
                submodules) doesn't have a registered grad sampler function.
        """
        super().__init__(
            m,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            strict=strict,
            force_functorch=force_functorch
        )
        self.group_size=group_size
        self.target_module=torch.nn.modules.linear.Linear

    def capture_backprops_hook(
        self,
        module: nn.Module,
        _forward_input: torch.Tensor,
        forward_output: torch.Tensor,
        loss_reduction: str,
        batch_first: bool,
    ):
        """
        Computes per sample gradients given the current backprops and activations
        stored by the associated forward hook. Computed per sample gradients are
        stored in ``grad_sample`` field in each parameter.

        For non-recurrent layers the process is straightforward: for each
        ``loss.backward()`` call this hook will be called exactly one. For recurrent
        layers, however, this is more complicated and the hook will be called multiple
        times, while still processing the same batch of data.

        For this reason we first accumulate the gradients from *the same batch* in
        ``p._current_grad_sample`` and then, when we detect the end of a full backward
        pass - we store accumulated result on ``p.grad_sample``.

        From there, ``p.grad_sample`` could be either a Tensor or a list of Tensors,
        if accumulated over multiple batches

        Args:
            module: nn.Module,
            _forward_input: torch.Tensor,
            forward_output: torch.Tensor,
            loss_reduction: str,
            batch_first: bool,
        """
        if not self.hooks_enabled:
            return

        backprops = forward_output[0].detach()

        ### adjust to max_group_len
        if (not hasattr(module, "max_batch_len")) and hasattr(module, "activations"):
            module.max_batch_len = _get_batch_size(
                module=module,
                batch_dim=0 if batch_first or type(module) is RNNLinear else 1,
            ) // self.group_size
        ###
        activations, backprops = self.rearrange_grad_samples(
            module=module,
            backprops=backprops,
            loss_reduction=loss_reduction,
            batch_first=batch_first,
        )

        if not self.force_functorch and type(module) in self.GRAD_SAMPLERS:
            grad_sampler_fn = self.GRAD_SAMPLERS[type(module)]
        else:
            grad_sampler_fn = ft_compute_per_sample_gradient

        if type(module) == self.target_module:
            grad_samples = grad_sampler_fn(module, activations, backprops, self.group_size)
        else:
            grad_samples = grad_sampler_fn(module, activations, backprops)

        for param, gs in grad_samples.items():
            create_or_accumulate_grad_sample(
                param=param, grad_sample=gs, max_batch_len=module.max_batch_len
            )

        # Detect end of current batch processing and switch accumulation
        # mode from sum to stacking. Used for RNNs and tied parameters
        # (See #417 for details)
        for _, p in trainable_parameters(module):
            p._forward_counter -= 1
            if p._forward_counter == 0:
                promote_current_grad_sample(p)
            
            if importlib.metadata.version('opacus') > '1.4.0':
                if not self.grad_accumulation_allowed:
                    if isinstance(p.grad_sample, list) and len(p.grad_sample) > 1:
                        raise ValueError(
                            "Poisson sampling is not compatible with grad accumulation. "
                            "You need to call optimizer.step() after every forward/backward pass "
                            "or consider using BatchMemoryManager"
                        )

        if len(module.activations) == 0:
            if hasattr(module, "max_batch_len"):
                del module.max_batch_len

    def add_hooks(
        self,
        *,
        loss_reduction: str = "mean",
        batch_first: bool = True,
        force_functorch: bool = False,
    ) -> None:
        """
        Adds hooks to model to save activations and backprop values.
        The hooks will
        1. save activations into param.activations during forward pass
        2. compute per-sample gradients in params.grad_sample during backward pass.
        Call ``remove_hooks(model)`` to disable this.

        Args:
            model: the model to which hooks are added
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            force_functorch: If set to ``True``, will use functorch to compute all per sample gradients.
                Otherwise, functorch will be used only for layers without registered grad sampler methods.
        """
        if hasattr(self._module, "autograd_grad_sample_hooks"):
            raise ValueError("Trying to add hooks twice to the same model")
        else:
            self._module.autograd_grad_sample_hooks = []
            self.autograd_grad_sample_hooks = self._module.autograd_grad_sample_hooks

        for module in self.iterate_submodules(self._module):
            # Do not add hooks to DPRNN, DPLSTM or DPGRU as the hooks are handled by the `RNNLinear`
            if type(module) in [DPRNN, DPLSTM, DPGRU]:
                continue

            if force_functorch or not type(module) in self.GRAD_SAMPLERS:
                prepare_layer(module, batch_first=batch_first)

            self.autograd_grad_sample_hooks.append(
                module.register_forward_hook(self.capture_activations_hook)
            )

            self.autograd_grad_sample_hooks.append(
                module.register_full_backward_hook(
                    partial(
                        self.capture_backprops_hook,
                        loss_reduction=loss_reduction,
                        batch_first=batch_first,
                    )
                )
            )

        self.enable_hooks()


from opt_einsum import contract
from opacus.grad_sample.utils import register_grad_sampler


@register_grad_sampler(nn.Linear)
def compute_linear_grad_sample_group(
    layer: nn.Linear, activations: List[torch.Tensor], backprops: torch.Tensor, tuple_size: int = -1
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per loss gradients aggregated within a contrastive group for ``nn.Linear`` layer

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    activations = activations[0] #shape [batch_size*tuple_size, len_seq, dim]
    ret = {}
    stack_size = torch.Size([tuple_size,-1])
    if layer.weight.requires_grad:
        # [batch_size*tuple_size, len_seq, dim] -> [tuple_size, batch_size, len_seq, dim] -> [batch_size, tuple_size*len_seq, dim]
        gs = contract("nm...i,nm...j->mij", backprops.view(stack_size+backprops.shape[1:]), activations.view(stack_size+activations.shape[1:])) if tuple_size >0 else contract("n...i,n...j->nij", backprops, activations) 
        ret[layer.weight] = gs
    if layer.bias is not None and layer.bias.requires_grad:
        ret[layer.bias] = contract("nm...k->mk", backprops.view(stack_size+backprops.shape[1:])) if tuple_size>0 else contract("n...k->nk", backprops)
    return ret