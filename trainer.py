# adapted from https://github.com/huggingface/transformers/blob/v4.39.3/src/transformers/trainer.py

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import torch
import torch.nn as nn
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach

### custumized private trainer ###
TRAINER_STATE_NAME = "trainer_state.json"

from transformers_support import forward_swapper

import torch.distributed as dist
from torch.utils.data import Dataset, RandomSampler
from opacus.utils.batch_memory_manager import BatchMemoryManager

import math, sys, time, shutil
import importlib.metadata
from packaging import version

from transformers.trainer_utils import (
    HPSearchBackend,
    TrainOutput,
    speed_metrics,
    has_length,
)

from transformers.trainer_pt_utils import (
    get_model_param_count,
    get_dataloader_sampler,
)

from transformers.debug_utils import DebugOption, DebugUnderflowOverflow

from transformers.training_args import ParallelMode

from transformers.trainer_callback import (
    TrainerState,
)

from transformers.utils import (
    is_apex_available,
    is_sagemaker_mp_enabled,
    is_accelerate_available,
    is_torch_xla_available,
    is_peft_available,
    logging,
    ADAPTER_WEIGHTS_NAME,
)

from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.integrations.deepspeed import deepspeed_init, deepspeed_load_checkpoint, is_deepspeed_available
from transformers.integrations import (
    hp_params,
)

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.spmd as xs
    import torch_xla.runtime as xr

if is_apex_available():
    from apex import amp

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from transformers.trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_peft_available():
    from peft import PeftModel

def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        # Here we also check if the model is an instance of `PeftMixedModel` introduced in peft>=0.7.0: https://github.com/huggingface/transformers/pull/28321
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False

if is_accelerate_available():
    from accelerate import Accelerator, skip_first_batches
    from accelerate import __version__ as accelerate_version
    from accelerate.utils import (
        DistributedDataParallelKwargs,
        DistributedType,
        GradientAccumulationPlugin,
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    DATA_SAMPLERS = [RandomSampler]
    if version.parse(accelerate_version) > version.parse("0.23.0"):
        from accelerate.data_loader import SeedableRandomSampler

        DATA_SAMPLERS += [SeedableRandomSampler]

    if is_deepspeed_available():
        from accelerate.utils import DeepSpeedSchedulerWrapper

logger = logging.get_logger(__name__)
logger.setLevel(logging.INFO)


def get_weighted_trainer(pos_weight, neg_weight):
    
    class _WeightedBCELossTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            # forward pass
            outputs = model(**inputs, output_hidden_states=True)
            logits = outputs.get("logits")
            # compute custom loss (suppose one has 3 labels with different weights)
            loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([neg_weight, pos_weight], device=labels.device, dtype=logits.dtype))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss
    return _WeightedBCELossTrainer

def get_galm_trainer(private=False):
    class _GaLMTrainer(Trainer):
        def _save(self, output_dir: Optional[str] = None):
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            logger.info("Saving model checkpoint to %s", output_dir)
            self.model.save(output_dir)

        def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
            if model is None:
                model = self.model_wrapped.lm if is_sagemaker_mp_enabled() else self.model.lm

            if is_peft_available() and isinstance(model, PeftModel):
                # If train a model using PEFT & LoRA, assume that adapter have been saved properly.
                if hasattr(model, "active_adapter") and hasattr(model, "load_adapter"):
                    if os.path.exists(resume_from_checkpoint):
                        model.load_adapter(resume_from_checkpoint, model.active_adapter, is_trainable=True)
                    else:
                        logger.warning(
                            "The intermediate checkpoints of PEFT may not be saved correctly, "
                            f"consider using a custom callback to save {ADAPTER_WEIGHTS_NAME} in corresponding saving folders. "
                            "Check some examples here: https://github.com/huggingface/peft/issues/96"
                        )
                else:
                    logger.warning("Could not load adapter model, make sure to have `peft>=0.3.0` installed")
            else:
                return super()._load_from_checkpoint(resume_from_checkpoint, model=model)

        def _prepare_inputs(
            self,
            inputs: Tuple[Dict[str, Union[torch.Tensor, Any]], ...]
        ) -> List[Dict[str, Union[torch.Tensor, Any]]]:
            prepared = []
            for x in inputs:
                if isinstance(x, torch.Tensor):
                    prepared.append(x.to(self.args.device))
                else:
                    prepared.append(super()._prepare_inputs(x))
            return prepared
    
        def compute_loss(self, model, inputs, return_outputs=False):
            if len(inputs) == 2:
                query, key = inputs
                outputs = model(query, key)
            else:
                query, key, key_neg = inputs
                outputs = model(query, key, key_neg)
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs[0]

            return (loss, outputs) if return_outputs else loss
        
        def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            """
            Perform an evaluation step on `model` using `inputs`.

            Subclass and override to inject custom behavior.

            Args:
                model (`nn.Module`):
                    The model to evaluate.
                inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.

                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument `labels`. Check your model's documentation for all accepted arguments.
                prediction_loss_only (`bool`):
                    Whether or not to return the loss only.
                ignore_keys (`List[str]`, *optional*):
                    A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                    gathering predictions.

            Return:
                Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
                logits and labels (each being optional).
            """
            has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
            # For CLIP-like models capable of returning loss values.
            # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
            # is `True` in `model.forward`.
            # return_loss = inputs.get("return_loss", None)
            # if return_loss is None:
            #     return_loss = self.can_return_loss
            return_loss = True
            loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            if has_labels or loss_without_labels:
                labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
                if len(labels) == 1:
                    labels = labels[0]
            else:
                labels = None

            with torch.no_grad():
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

            if prediction_loss_only:
                return (loss, None, None)

            logits = nested_detach(logits)
            if len(logits) == 1:
                logits = logits[0]

            return (loss, logits, labels)
        
    class _PvGaLMTrainer(_GaLMTrainer):
        def __init__(self, *args, **kwargs):
            super(_PvGaLMTrainer, self).__init__(*args, **kwargs)
            if "bert" in self.model.model_args.model_type:
                for p in self.model.lm.pooler.parameters():
                    # Fix the pooler grad_sample issue
                    p.requires_grad = False
                if not isinstance(self.model.lm, PeftModel):
                    # Fix the position embeddings broadcast issue.
                    forward_swapper(module=self.model.lm)

        def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
        ):  
            self.accelerator.free_memory()
            self._train_batch_size = batch_size
            if self.args.auto_find_batch_size:
                if self.state.train_batch_size != self._train_batch_size:
                    from accelerate.utils import release_memory

                    (self.model_wrapped,) = release_memory(self.model_wrapped)
                    self.model_wrapped = self.model

                    # Check for DeepSpeed *after* the initial pass and modify the config
                    if self.is_deepspeed_enabled:
                        # Temporarily unset `self.args.train_batch_size`
                        original_bs = self.args.per_device_train_batch_size
                        self.args.per_device_train_batch_size = self._train_batch_size // max(1, self.args.n_gpu)
                        self.propagate_args_to_deepspeed(True)
                        self.args.per_device_train_batch_size = original_bs
                self.state.train_batch_size = self._train_batch_size
            logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
            # Data loader and number of training steps
            train_dataloader = self.get_train_dataloader()
            if self.is_fsdp_xla_v2_enabled:
                train_dataloader = tpu_spmd_dataloader(train_dataloader)

            # Setting up training control variables:
            # number of training epochs: num_train_epochs
            # number of training steps per epoch: num_update_steps_per_epoch
            # total number of training steps to execute: max_steps
            total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

            len_dataloader = None
            num_train_tokens = None
            if has_length(train_dataloader):
                len_dataloader = len(train_dataloader)
                num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
                num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
                num_examples = self.num_examples(train_dataloader)
                if args.max_steps > 0:
                    max_steps = args.max_steps
                    num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                        args.max_steps % num_update_steps_per_epoch > 0
                    )
                    # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                    # the best we can do.
                    num_train_samples = args.max_steps * total_train_batch_size
                    if args.include_tokens_per_second:
                        num_train_tokens = (
                            self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                        )
                else:
                    max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                    num_train_epochs = math.ceil(args.num_train_epochs)
                    num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                    if args.include_tokens_per_second:
                        num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
            elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
                max_steps = args.max_steps
                # Setting a very large number of epochs so we go as many times as necessary over the iterator.
                num_train_epochs = sys.maxsize
                num_update_steps_per_epoch = max_steps
                num_examples = total_train_batch_size * args.max_steps
                num_train_samples = args.max_steps * total_train_batch_size
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
            else:
                raise ValueError(
                    "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                    f" {args.max_steps}"
                )

            if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
                if self.args.n_gpu > 1:
                    # nn.DataParallel(model) replicates the model, creating new variables and module
                    # references registered here no longer work on other gpus, breaking the module
                    raise ValueError(
                        "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                        " (torchrun or torch.distributed.launch (deprecated))."
                    )
                else:
                    debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

            delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

            # We need to reset the scheduler, as its parameters may be different on subsequent calls
            if self._created_lr_scheduler:
                self.lr_scheduler = None
                self._created_lr_scheduler = False

            if self.is_deepspeed_enabled:
                self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

            if not delay_optimizer_creation:
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)

            self.state = TrainerState()
            self.state.is_hyper_param_search = trial is not None
            self.state.train_batch_size = self._train_batch_size

            # Compute absolute values for logging, eval, and save if given as ratio
            if args.logging_steps is not None:
                if args.logging_steps < 1:
                    self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
                else:
                    self.state.logging_steps = args.logging_steps
            if args.eval_steps is not None:
                if args.eval_steps < 1:
                    self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
                else:
                    self.state.eval_steps = args.eval_steps
            if args.save_steps is not None:
                if args.save_steps < 1:
                    self.state.save_steps = math.ceil(max_steps * args.save_steps)
                else:
                    self.state.save_steps = args.save_steps

            # Activate gradient checkpointing if needed
            if args.gradient_checkpointing:
                if args.gradient_checkpointing_kwargs is None:
                    gradient_checkpointing_kwargs = {}
                else:
                    gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

            model = self._wrap_model(self.model_wrapped)

            # as the model is wrapped, don't use `accelerator.prepare`
            # this is for unhandled cases such as
            # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
            use_accelerator_prepare = True if model is self.model else False

            if delay_optimizer_creation:
                if use_accelerator_prepare:
                    self._fsdp_qlora_plugin_updates()
                    self.model = self.accelerator.prepare(self.model)
                self.create_optimizer_and_scheduler(num_training_steps=max_steps)

            # prepare using `accelerator` prepare
            if use_accelerator_prepare:
                self.model.train()
                if hasattr(self.lr_scheduler, "step"):
                    if self.use_apex:
                        model = self.accelerator.prepare(self.model)
                    else:
                        model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
                else:
                    # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                    model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                        self.model, self.optimizer, self.lr_scheduler
                    )

            if self.is_fsdp_enabled:
                self.model = self.model_wrapped = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

            # ckpt loading
            if resume_from_checkpoint is not None:
                if self.is_deepspeed_enabled:
                    deepspeed_load_checkpoint(
                        self.model_wrapped, resume_from_checkpoint, load_module_strict=not _is_peft_model(self.model)
                    )
                elif is_sagemaker_mp_enabled() or self.is_fsdp_enabled:
                    self._load_from_checkpoint(resume_from_checkpoint, self.model_wrapped)

            # Check if saved optimizer or scheduler states exist
            self._load_optimizer_and_scheduler(resume_from_checkpoint)

            # important: at this point:
            # self.model         is the Transformers Model
            # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
            # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

            """
            Add Privacy Engine
            """
            if hasattr(self.args, "privacy_engine"):
                if self.args.noise_scale >= 0:
                    model, self.optimizer, train_dataloader = self.args.privacy_engine.make_private(
                    module=model,
                    optimizer=self.optimizer,
                    data_loader=train_dataloader,
                    noise_multiplier=self.args.noise_scale,
                    max_grad_norm=self.args.max_pgrad_norm,
                    )
                    if self.args.dp_type == "node":
                        self.args.dp_optimizer = self.optimizer
                elif self.args.epsilon > 0:
                    model, self.optimizer, train_dataloader = self.args.privacy_engine.make_private_with_epsilon(
                        module=model,
                        optimizer=self.optimizer,
                        data_loader=train_dataloader,
                        target_delta=self.args.delta,
                        target_epsilon=self.args.epsilon,
                        epochs=num_train_epochs,
                        max_grad_norm=self.args.max_pgrad_norm,
                    )
                else:
                    raise ValueError("Privacy Engine requires either epsilon or noise_scale to be set.")
                
                logger.info(f"  Using Private Dense Trainer!") 
                logger.info(f"  Privacy Settings: epsilon {self.args.epsilon} delta {self.args.delta} ")
                logger.info(f"  Privacy Parameters: max_grad_norm {self.args.max_pgrad_norm} noise_scale {self.optimizer.noise_multiplier}")

            # Train!
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {num_examples:,}")
            logger.info(f"  Num Epochs = {num_train_epochs:,}")
            logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
            if self.args.per_device_train_batch_size != self._train_batch_size:
                logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
            logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {max_steps:,}")
            logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

            self.state.epoch = 0
            start_time = time.time()
            epochs_trained = 0
            steps_trained_in_current_epoch = 0
            steps_trained_progress_bar = None

            # Check if continuing training from a checkpoint
            if resume_from_checkpoint is not None and os.path.isfile(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            ):
                self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
                epochs_trained = self.state.global_step // num_update_steps_per_epoch
                if not args.ignore_data_skip:
                    steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                    steps_trained_in_current_epoch *= args.gradient_accumulation_steps
                else:
                    steps_trained_in_current_epoch = 0

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info(f"  Continuing training from epoch {epochs_trained}")
                logger.info(f"  Continuing training from global step {self.state.global_step}")
                if not args.ignore_data_skip:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch."
                    )

            # Update the references
            self.callback_handler.model = self.model
            self.callback_handler.optimizer = self.optimizer
            self.callback_handler.lr_scheduler = self.lr_scheduler
            self.callback_handler.train_dataloader = train_dataloader
            if self.hp_name is not None and self._trial is not None:
                # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
                # parameter to Train when using DDP.
                self.state.trial_name = self.hp_name(self._trial)
            if trial is not None:
                assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
                self.state.trial_params = hp_params(assignments)
            else:
                self.state.trial_params = None
            # This should be the same if the state has been saved but in case the training arguments changed, it's safer
            # to set this after the load.
            self.state.max_steps = max_steps
            self.state.num_train_epochs = num_train_epochs
            self.state.is_local_process_zero = self.is_local_process_zero()
            self.state.is_world_process_zero = self.is_world_process_zero()

            # tr_loss is a tensor to avoid synchronization of TPUs through .item()
            tr_loss = torch.tensor(0.0).to(args.device)
            # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
            self._total_loss_scalar = 0.0
            self._globalstep_last_logged = self.state.global_step
            model.zero_grad()
            grad_norm: Optional[float] = None

            self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

            # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
            if not args.ignore_data_skip:
                for epoch in range(epochs_trained):
                    sampler = get_dataloader_sampler(train_dataloader)
                    sampler_kinds = [RandomSampler]
                    if version.parse(accelerate_version) > version.parse("0.23.0"):
                        sampler_kinds.append(SeedableRandomSampler)
                    is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                    if not is_random_sampler:
                        # We just need to begin an iteration to create the randomization of the sampler.
                        for _ in train_dataloader:
                            break
                    else:
                        # Otherwise we need to call the whooooole sampler cause there is some random operation added
                        # AT THE VERY END!
                        sampler = sampler if sampler is not None else []
                        _ = list(sampler)

            total_batched_samples = 0
            for epoch in range(epochs_trained, num_train_epochs):
                epoch_iterator = train_dataloader
                if hasattr(epoch_iterator, "set_epoch"):
                    epoch_iterator.set_epoch(epoch)

                # Reset the past mems state at the beginning of each epoch if necessary.
                if args.past_index >= 0:
                    self._past = None

                steps_in_epoch = (
                    len(epoch_iterator)
                    if len_dataloader is not None
                    else args.max_steps * args.gradient_accumulation_steps
                )
                self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

                if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                    self._load_rng_state(resume_from_checkpoint)

                rng_to_sync = False
                steps_skipped = 0
                if steps_trained_in_current_epoch > 0:
                    epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                    steps_skipped = steps_trained_in_current_epoch
                    steps_trained_in_current_epoch = 0
                    rng_to_sync = True

                with BatchMemoryManager(data_loader=epoch_iterator,
                        max_physical_batch_size=self.args.max_physical_batch_size,
                        optimizer=self.optimizer) as memory_safe_data_loader:
                    step = -1
                    steps_batch_accum = 0
                    for step, inputs in enumerate(memory_safe_data_loader):
                        self.optimizer.zero_grad()
                        total_batched_samples += 1

                        if self.args.include_num_input_tokens_seen:
                            main_input_name = getattr(self.model, "main_input_name", "input_ids")
                            if main_input_name not in inputs:
                                logger.warning(
                                    "Tried to track the number of tokens seen, however the current model is "
                                    "not configured properly to know what item is the input. To fix this, add "
                                    "a `main_input_name` attribute to the model class you are using."
                                )
                            else:
                                self.state.num_input_tokens_seen += self.accelerator.gather(inputs[main_input_name]).numel()
                        if rng_to_sync:
                            self._load_rng_state(resume_from_checkpoint)
                            rng_to_sync = False

                        # Skip past any already trained steps if resuming training
                        if steps_trained_in_current_epoch > 0:
                            steps_trained_in_current_epoch -= 1
                            if steps_trained_progress_bar is not None:
                                steps_trained_progress_bar.update(1)
                            if steps_trained_in_current_epoch == 0:
                                self._load_rng_state(resume_from_checkpoint)
                            continue
                        elif steps_trained_progress_bar is not None:
                            steps_trained_progress_bar.close()
                            steps_trained_progress_bar = None

                        if step % args.gradient_accumulation_steps == 0:
                            self.control = self.callback_handler.on_step_begin(args, self.state, self.control)
                        
                        # checkpoint & autocast
                        with self.accelerator.accumulate(model):
                            tr_loss_step = self.training_step(model, inputs)

                        if (
                            args.logging_nan_inf_filter
                            and not is_torch_xla_available()
                            and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                        ):
                            # if loss is nan or inf simply add the average of previous logged losses
                            tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                        else:
                            if tr_loss.device != tr_loss_step.device:
                                raise ValueError(
                                    f"Calculated loss must be on the original device: {tr_loss.device} but device in use is {tr_loss_step.device}"
                                )
                            tr_loss += tr_loss_step

                        self.current_flos += float(self.floating_point_ops(inputs))

                        is_last_step_and_steps_less_than_grad_acc = (
                            steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                        )

                        if (
                            total_batched_samples % args.gradient_accumulation_steps == 0
                            or
                            # last step in epoch but step is always smaller than gradient_accumulation_steps
                            is_last_step_and_steps_less_than_grad_acc
                        ):
                            # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                            # in accelerate. So, explicitly enable sync gradients to True in that case.
                            if is_last_step_and_steps_less_than_grad_acc:
                                self.accelerator.gradient_state._set_sync_gradients(True)

                            ### Gradient clipping overrided by privacy setting
                            if args.max_grad_norm is not None and args.max_grad_norm > 0 and not hasattr(self.args, "privacy_engine"):
                                # deepspeed does its own clipping

                                if is_sagemaker_mp_enabled() and args.fp16:
                                    _grad_norm = self.optimizer.clip_master_grads(args.max_grad_norm)
                                elif self.use_apex:
                                    # Revert to normal clipping otherwise, handling Apex or full precision
                                    _grad_norm = nn.utils.clip_grad_norm_(
                                        amp.master_params(self.optimizer),
                                        args.max_grad_norm,
                                    )
                                else:
                                    _grad_norm = self.accelerator.clip_grad_norm_(
                                        model.parameters(),
                                        args.max_grad_norm,
                                    )

                                if (
                                    is_accelerate_available()
                                    and self.accelerator.distributed_type == DistributedType.DEEPSPEED
                                ):
                                    grad_norm = model.get_global_grad_norm()
                                    # In some cases the grad norm may not return a float
                                    if hasattr(grad_norm, "item"):
                                        grad_norm = grad_norm.item()
                                else:
                                    grad_norm = _grad_norm

                            # Optimizer step
                            if self.optimizer.step() == 'skipped':
                                optimizer_was_run = False
                                steps_batch_accum += 1
                            else:
                                optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                            if optimizer_was_run:
                                # Delay optimizer scheduling until metrics are generated
                                if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                                    self.lr_scheduler.step()

                                # model.zero_grad()
                                self.state.global_step += 1
                                self.state.epoch = epoch + (step + 1 + steps_skipped-steps_batch_accum) / steps_in_epoch
                                self.control = self.callback_handler.on_step_end(args, self.state, self.control)
                                if self.state.global_step < self.args.start_eval*(64//self.args.per_device_train_batch_size):
                                    self.control.should_evaluate=False
                                    self.control.should_save=False
                                self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)
                        else:
                            self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                        if self.control.should_epoch_stop or self.control.should_training_stop:
                            # PyTorch/XLA relies on the data loader to insert the mark_step for
                            # each step. Since we are breaking the loop early, we need to manually
                            # insert the mark_step here.
                            if is_torch_xla_available():
                                xm.mark_step()
                            break
                if step < 0:
                    logger.warning(
                        "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                        f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                        f" num_steps ({max_steps}) higher than the number of available samples."
                    )
                    self.control.should_training_stop = True

                self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
                self._maybe_log_save_evaluate(tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval)

                if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                    if is_torch_xla_available():
                        # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                        xm.master_print(met.metrics_report())
                    else:
                        logger.warning(
                            "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                            "configured. Check your training configuration if this is unexpected."
                        )
                if self.control.should_training_stop:
                    break

            if args.past_index and hasattr(self, "_past"):
                # Clean the state at the end of training
                delattr(self, "_past")

            logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
            if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
                # Wait for everyone to get here so we are sure the model has been saved by process 0.
                if is_torch_xla_available():
                    xm.rendezvous("load_best_model_at_end")
                elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                    dist.barrier()
                elif is_sagemaker_mp_enabled():
                    smp.barrier()

                self._load_best_model()

            # add remaining tr_loss
            self._total_loss_scalar += tr_loss.item()
            effective_global_step = max(self.state.global_step, 0.001)  # Avoid ZeroDivisionError
            train_loss = self._total_loss_scalar / effective_global_step

            metrics = speed_metrics(
                "train",
                start_time,
                num_samples=num_train_samples,
                num_steps=self.state.max_steps,
                num_tokens=num_train_tokens,
            )
            self.store_flos()
            metrics["total_flos"] = self.state.total_flos
            metrics["train_loss"] = train_loss

            self.is_in_train = False

            self._memory_tracker.stop_and_update_metrics(metrics)

            self.log(metrics)

            run_dir = self._get_output_dir(trial)
            checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

            # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
            if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
                for checkpoint in checkpoints_sorted:
                    if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                        logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                        shutil.rmtree(checkpoint)

            self.control = self.callback_handler.on_train_end(args, self.state, self.control)

            # Wait for the checkpoint to be uploaded.
            self._finish_current_push()

            # After training we make sure to retrieve back the original forward pass method
            # for the embedding layer by removing the forward post hook.
            if self.neftune_noise_alpha is not None:
                self._deactivate_neftune(self.model)

            return TrainOutput(self.state.global_step, train_loss, metrics)

        def evaluate(
            self,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
        ) -> Dict[str, float]:
            """
            Run evaluation and returns metrics.

            The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
            (pass it to the init `compute_metrics` argument).

            You can also subclass and override this method to inject custom behavior.

            Args:
                eval_dataset (Union[`Dataset`, Dict[str, `Dataset`]), *optional*):
                    Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                    not accepted by the `model.forward()` method are automatically removed. If it is a dictionary, it will
                    evaluate on each dataset, prepending the dictionary key to the metric name. Datasets must implement the
                    `__len__` method.

                    <Tip>

                    If you pass a dictionary with names of datasets as keys and datasets as values, evaluate will run
                    separate evaluations on each dataset. This can be useful to monitor how training affects other
                    datasets or simply to get a more fine-grained evaluation.
                    When used with `load_best_model_at_end`, make sure `metric_for_best_model` references exactly one
                    of the datasets. If you, for example, pass in `{"data1": data1, "data2": data2}` for two datasets
                    `data1` and `data2`, you could specify `metric_for_best_model="eval_data1_loss"` for using the
                    loss on `data1` and `metric_for_best_model="eval_data1_loss"` for the loss on `data2`.

                    </Tip>

                ignore_keys (`List[str]`, *optional*):
                    A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                    gathering predictions.
                metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                    An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                    "eval_bleu" if the prefix is "eval" (default)

            Returns:
                A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
                dictionary also contains the epoch number which comes from the training state.
            """
            # handle multiple eval datasets
            eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
            if isinstance(eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, _eval_dataset in eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=_eval_dataset,
                        ignore_keys=ignore_keys,
                        metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
                return metrics

            # memory metrics - must set up as early as possible
            self._memory_tracker.start()

            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            if self.is_fsdp_xla_v2_enabled:
                eval_dataloader = tpu_spmd_dataloader(eval_dataloader)

            start_time = time.time()

            eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

            total_batch_size = self.args.eval_batch_size * self.args.world_size
            if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
                start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
            output.metrics.update(
                speed_metrics(
                    metric_key_prefix,
                    start_time,
                    num_samples=output.num_samples,
                    num_steps=math.ceil(output.num_samples / total_batch_size),
                )
            )

            output.metrics['ɛ'] = self.args.privacy_engine.get_epsilon(self.args.delta) if (hasattr(self.args, "privacy_engine") and self.args.noise_scale >= 0.1) else 0.0

            self.log(output.metrics)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

            self._memory_tracker.stop_and_update_metrics(output.metrics)

            return output.metrics

    return _PvGaLMTrainer if private else _GaLMTrainer
