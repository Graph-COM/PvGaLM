from opacus.privacy_engine import PrivacyEngine
from opacus.optimizers.adaclipoptimizer import AdaClipDPOptimizer
from opacus.optimizers.ddp_perlayeroptimizer import (
    DistributedPerLayerOptimizer,
    SimpleDistributedPerLayerOptimizer,
)
from opacus.optimizers.ddpoptimizer import DistributedDPOptimizer
from opacus.optimizers.perlayeroptimizer import DPPerLayerOptimizer
from opacus.optimizers import DPOptimizer
from opacus.grad_sample import AbstractGradSampleModule

from opacus.grad_sample.grad_sample_module import GradSampleModule
from opacus.grad_sample.gsm_exp_weights import GradSampleModuleExpandedWeights
from opacus.grad_sample.gsm_no_op import GradSampleModuleNoOp

from dpgrl.optimizer import GDPOptimizer
from typing import IO, Any, BinaryIO, Dict, List, Optional, Tuple, Union, Type
from torch import nn, optim
from dpgrl.grad_sample_module import GDPGradSampleModule
import pdb

class GPrivacyEngine(PrivacyEngine):
    def __init__(self, *, accountant: str = "prv", secure_mode: bool = False, preclip=0, neg_k=0, dp_type=None):
        """

        Args:
            accountant: Accounting mechanism. Currently supported:
                - rdp (:class:`~opacus.accountants.RDPAccountant`)
                - gdp (:class:`~opacus.accountants.GaussianAccountant`)
                - prv (:class`~opacus.accountants.PRVAccountant`)
            secure_mode: Set to ``True`` if cryptographically strong DP guarantee is
                required. ``secure_mode=True`` uses secure random number generator for
                noise and shuffling (as opposed to pseudo-rng in vanilla PyTorch) and
                prevents certain floating-point arithmetic-based attacks.
                See :meth:`~opacus.optimizers.optimizer._generate_noise` for details.
                When set to ``True`` requires ``torchcsprng`` to be installed
        """
        super().__init__(accountant=accountant, secure_mode=secure_mode)
        self.preclip = preclip
        self.group_size = neg_k+2
        self.dp_type = dp_type

    def _prepare_optimizer(
        self,
        optimizer: optim.Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        expected_batch_size: int,
        loss_reduction: str = "mean",
        distributed: bool = False,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode="hooks",
    ) -> DPOptimizer | GDPOptimizer:
        if isinstance(optimizer, DPOptimizer) or isinstance(optimizer, GDPOptimizer):
            optimizer = optimizer.original_optimizer

        generator = None
        if self.secure_mode:
            generator = self.secure_rng
        elif noise_generator is not None:
            generator = noise_generator

        optim_class = self._get_optimizer_class(
            clipping=clipping,
            distributed=distributed,
            grad_sample_mode=grad_sample_mode,
            dp_type = self.dp_type,
        )

        if optim_class == GDPOptimizer:
            return optim_class(
                optimizer=optimizer,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                expected_batch_size=expected_batch_size,
                loss_reduction=loss_reduction,
                generator=generator,
                secure_mode=self.secure_mode,
                preclip = self.preclip,
                neg_k = self.group_size-2,
                dp_type = self.dp_type,
            )
        else:
            return optim_class(
                optimizer=optimizer,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                expected_batch_size=expected_batch_size,
                loss_reduction=loss_reduction,
                generator=generator,
                secure_mode=self.secure_mode,
            )
    
    def _get_optimizer_class(self, clipping: str, distributed: bool, grad_sample_mode: str = None, dp_type=None):
        if clipping == "flat" and distributed is False:
            if dp_type is None:
                return DPOptimizer
            else:
                if dp_type in ['node', 'edge']:
                    return GDPOptimizer
                else:
                    raise NotImplementedError(f"Unexpected dp_type: {dp_type} for GDPOptimizer")
        elif clipping == "flat" and distributed is True:
            return DistributedDPOptimizer
        elif clipping == "per_layer" and distributed is False:
            return DPPerLayerOptimizer
        elif clipping == "per_layer" and distributed is True:
            if grad_sample_mode == "hooks":
                return DistributedPerLayerOptimizer
            elif grad_sample_mode == "ew":
                return SimpleDistributedPerLayerOptimizer
            else:
                raise ValueError(f"Unexpected grad_sample_mode: {grad_sample_mode}")
        elif clipping == "adaptive" and distributed is False:
            return AdaClipDPOptimizer
        raise ValueError(
            f"Unexpected optimizer parameters. Clipping: {clipping}, distributed: {distributed}"
        )
    
    def _prepare_model(
        self,
        module: nn.Module,
        *,
        batch_first: bool = True,
        loss_reduction: str = "mean",
        grad_sample_mode: str = "hooks",
    ) -> AbstractGradSampleModule:
        # Ideally, validation should have been taken care of by calling
        # `get_compatible_module()`
        self.validate(module=module, optimizer=None, data_loader=None)

        # wrap
        if isinstance(module, AbstractGradSampleModule):
            if (
                module.batch_first != batch_first
                or module.loss_reduction != loss_reduction
                or type(module) is not get_gsm_class(grad_sample_mode)
            ):
                raise ValueError(
                    f"Pre-existing GradSampleModule doesn't match new arguments."
                    f"Got: module.batch_first: {module.batch_first}, module.loss_reduction: {module.loss_reduction}, type(module): {type(module)}"
                    f"Requested: batch_first:{batch_first}, loss_reduction: {loss_reduction}, grad_sample_mode: {grad_sample_mode} "
                    f"Please pass vanilla nn.Module instead"
                )

            return module
        else:
            return wrap_model(
                module,
                grad_sample_mode=grad_sample_mode,
                batch_first=batch_first,
                loss_reduction=loss_reduction,
                group_size=self.group_size,
            )
        
def wrap_model(model: nn.Module, grad_sample_mode: str, *args, **kwargs):
    cls = get_gsm_class(grad_sample_mode, group_gsm='group_size' in kwargs)
    if grad_sample_mode == "functorch":
        kwargs["force_functorch"] = True
    return cls(model, *args, **kwargs)
        
def get_gsm_class(grad_sample_mode: str, group_gsm=False) -> Type[AbstractGradSampleModule]:
    """
    Returns AbstractGradSampleModule subclass correspinding to the input mode.
    See README for detailed comparison between grad sample modes.

    :param grad_sample_mode:
    :return:
    """
    if grad_sample_mode in ["hooks", "functorch"]:
        if group_gsm:
            return GDPGradSampleModule
        else:
            return GradSampleModule
    elif grad_sample_mode == "ew":
        return GradSampleModuleExpandedWeights
    elif grad_sample_mode == "no_op":
        return GradSampleModuleNoOp
    else:
        raise ValueError(
            f"Unexpected grad_sample_mode: {grad_sample_mode}. "
            f"Allowed values: hooks, ew"
        )