import sys
import os
from copy import deepcopy

import traceback
import logging
import wandb

import torch
torch.set_float32_matmul_precision("high")

from transformers import AutoTokenizer, AutoConfig, set_seed
from transformers import TrainingArguments, HfArgumentParser, TrainerCallback
from peft import get_peft_model, LoraConfig, TaskType

from dataset import TrainCollator, TrainDataset, DPTrainDataset, EvalDataset
from arguments import DataArguments, ModelArguments, DenseTrainingArguments, PrivateTrainingArguments
from model import GaLMModel, DPGaLMModel
from trainer import get_galm_trainer
from utils import compute_metrics
from dpgrl.privacy_engine import GPrivacyEngine

logger = logging.getLogger(__name__)


class CustomCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

def get_lora_model(model_checkpoints, model, rank=4, alpha=16, lora_dropout=0.1, bias='none'):
    if 'bert' in model_checkpoints:
        peft_config = LoraConfig(
            target_modules=['query', 'key', 'value', 'dense'],
            r=rank, lora_alpha=alpha, lora_dropout=lora_dropout, bias=bias
        )
    elif model_checkpoints == 'mistralai/Mistral-7B-v0.1' or model_checkpoints == 'meta-llama/Llama-2-7b-hf': 
        peft_config = LoraConfig(
            target_modules=["q_proj", "v_proj"],
            r=rank, lora_alpha=alpha, lora_dropout=lora_dropout, bias=bias, 
    )
    else: 
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=rank, lora_alpha=alpha, lora_dropout=lora_dropout, bias=bias,
        )
    model.lm = get_peft_model(model.lm, peft_config)
    logger.info(f'Lora config: rank {rank}, alpha {alpha}, dropout {lora_dropout}, bias {bias}')
    model.lm.print_trainable_parameters()
    return model

def main():
    """
    Training function
    """
    train_private='prv' in sys.argv[0]

    if train_private: 
        parser = HfArgumentParser((ModelArguments, DataArguments, PrivateTrainingArguments))
        model_class = DPGaLMModel
        dtrain_class = DPTrainDataset
    else:
        parser = HfArgumentParser((ModelArguments, DataArguments, DenseTrainingArguments))
        model_class = GaLMModel
        dtrain_class = TrainDataset

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments
    
    if model_args.pdebug:
        wandb.init(mode="disabled")
                   
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != 0),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)

    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=1,
        cache_dir=model_args.cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        add_prefix_space=True,
        use_fast=False,
    )

    pt_model = model_class.build(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    if training_args.use_peft:
        model = get_lora_model(
            model_args.model_name_or_path,
            pt_model,
            rank=training_args.lora_rank,
            alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )
    else:
        model = pt_model

    if data_args.set_pad_id: 
        tokenizer.pad_token = tokenizer.eos_token
        model.lm.config.pad_token_id = model.lm.config.eos_token_id

    # move model to GPU device
    if model.lm.device.type != 'cuda':
        model.lm = model.lm.to('cuda')

    train_dataset = dtrain_class(tokenizer, data_args, shuffle_seed=training_args.seed, cache_dir=data_args.data_cache_dir or model_args.cache_dir, k=training_args.neg_k if train_private else None)
    eval_dataset = EvalDataset(tokenizer, data_args, shuffle_seed=training_args.seed, cache_dir=data_args.data_cache_dir or model_args.cache_dir) if data_args.eval_path is not None else None
    
    if train_private:
        assert training_args.gradient_accumulation_steps == 1
        training_args.delta = 1/len(train_dataset)
        training_args.privacy_engine = GPrivacyEngine(preclip=training_args.preclip, neg_k=training_args.neg_k, dp_type=training_args.dp_type)

    galm_trainer = get_galm_trainer(private=train_private)
    
    trainer = galm_trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=TrainCollator(
            tokenizer,
            max_len=data_args.max_len,
        ),
        compute_metrics=compute_metrics
    )
    trainer.add_callback(CustomCallback(trainer))
    train_dataset.trainer = trainer
    trainer.train()


if __name__ == "__main__":
    main()