import os
from dataclasses import dataclass, field
from typing import Optional, List
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    target_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained reranker target model"}
    )
    model_type: str = field(
        default=None,
        metadata={"help": "Name of the used model"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    # out projection
    add_pooler: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=768)

    # for Jax training
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one "
                    "of `[float32, float16, bfloat16]`. "
        },
    )

    # neighbor masking
    neighbor_mask_ratio: Optional[float] = field(
        default=0,
        metadata={
            "help": "The probability of neighbor to be masked/corrupted during link prediction/neighbor-enhanced mlm pretraining"
        }
    )

    pdebug: bool = field(default=False)

@dataclass
class DataArguments:
    data_dir: str = field(
        default=None, metadata={"help": "Path to data directory"}
    )
    train_dir: str = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    data_path: str = field(
        default=None, metadata={"help": "Path to the single data file"}
    )
    train_path: str = field(
        default=None, metadata={"help": "Path to single train file"}
    )
    eval_path: str = field(
        default=None, metadata={"help": "Path to eval file"}
    )
    test_path: str = field(
        default=None, metadata={"help": "Path to eval file"}
    )
    query_path: str = field(
        default=None, metadata={"help": "Path to query file"}
    )
    corpus_path: str = field(
        default=None, metadata={"help": "Path to corpus file"}
    )
    processed_data_path: str = field(
        default=None, metadata={"help": "Path to processed data directory"}
    )
    data_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the data downloaded from huggingface"}
    )
    dataset_name: str = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    passage_field_separator: str = field(default=' ')
    dataset_proc_num: int = field(
        default=12, metadata={"help": "number of proc used in dataset preprocess"}
    )
    hn_num: int = field(
        default=4, metadata={"help": "number of negatives used"}
    )
    positive_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first positive passage"})
    negative_passage_no_shuffle: bool = field(
        default=False, metadata={"help": "always use the first negative passages"}
    )

    encode_in_path: List[str] = field(default=None, metadata={"help": "Path to data to encode"})
    
    encode_is_qry: bool = field(default=False)
    save_trec: bool = field(default=False)
    encode_num_shard: int = field(default=1)
    encode_shard_index: int = field(default=0)

    max_len: int = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization for query. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )

    query_column_names: str = field(
        default="id,text",
        metadata={"help": "column names for the tsv data format"}
    )
    doc_column_names: str = field(
        default="id,title,text",
        metadata={"help": "column names for the tsv data format"}
    )

    # mlm pretrain
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    mlm_probability: Optional[float] = field(
        default=0.15,
        metadata={
            "help": "The probability of token to be masked/corrupted during Mask Language Modeling"
        }
    )

    # rerank
    pos_rerank_num: int = field(default=5)
    neg_rerank_num: int = field(default=15)

    # coarse-grained node classification
    class_num: int = field(default=10)

    set_pad_id: bool = field(default=False, metadata={"help": "set the pad id to 0"})

    def __post_init__(self):
        pass


@dataclass
class DenseTrainingArguments(TrainingArguments):
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})

    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)

    fix_bert: bool = field(default=False, metadata={"help": "fix BERT encoder during training or not"})

    mlm_loss: bool = field(default=False, metadata={"help": "use mlm loss or not"})
    mlm_weight: float = field(default=1, metadata={"help": "weight of mlm loss"})

    warmup_ratio: float = field(default=0.1)
    use_peft: bool = field(default=False, metadata={"help": "use PEFT or not"})
    quantization: bool = field(default=False, metadata={"help": "use quantization or not"})
    resume_training: bool = field(default=False, metadata={"help": "Resume training from a checkpoint"})

    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_rank: Optional[int] = field(default=8, metadata={"help": "lora rank"})
    lora_dropout: Optional[float] = field(default=0.1, metadata={"help": "lora dropout"})
    lora_bias: Optional[str] = field(default='none', metadata={"help": "Layers to add learnable bias"})



@dataclass
class PrivateTrainingArguments(TrainingArguments):
    negatives_x_device: bool = field(default=False, metadata={"help": "share negatives across devices"})
    do_encode: bool = field(default=False, metadata={"help": "run the encoding loop"})

    grad_cache: bool = field(default=False, metadata={"help": "Use gradient cache update"})
    gc_q_chunk_size: int = field(default=4)
    gc_p_chunk_size: int = field(default=32)

    fix_bert: bool = field(default=False, metadata={"help": "fix BERT encoder during training or not"})

    mlm_loss: bool = field(default=False, metadata={"help": "use mlm loss or not"})
    mlm_weight: float = field(default=1, metadata={"help": "weight of mlm loss"})

    hub_strategy: Optional[str] = field(
        default='end',
        metadata={
            "help": "Defines the scope of what is pushed to the Hub and when.r"
        }
    )

    optim: Optional[str] = field(
        default='adamw_torch',
        metadata={
            "help": "default optimizer"
        }
    )

    learning_rate: Optional[float] = field(
        default=1e-5,
        metadata={"help": "The initial learning rate for AdamW optimizer"}
    )

    warmup_ratio: float = field(default=0.1)

    start_eval: Optional[int] = field(
        default=0,
        metadata={"help": "step to start evaluation"}
    )

    early_stop: Optional[int] = field(
        default=10,
        metadata={"help": "step to stop training if no improvement"}
    )

    weight_decay: Optional[float] = field(
        default=0.0,
        metadata={"help": "Weight decay"}
    )

    seed: Optional[int] = field(
        default=2024,
        metadata={"help": "random seed"}
    )

    lora_alpha: Optional[int] = field(default=16, metadata={"help": "lora alpha"})
    lora_rank: Optional[int] = field(default=8, metadata={"help": "lora rank"})
    lora_dropout: Optional[float] = field(default=0, metadata={"help": "lora dropout"})
    lora_bias: Optional[str] = field(default='none', metadata={"help": "Layers to add learnable bias"})

    epsilon: Optional[float] = field(
        default=-1.0,
        metadata={
            "help": "the privacy budget epsilon"
        }
    )

    max_grad_norm: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "the maximum magnitude of L2 norms to perform gradient clipping"
        }
    )

    max_pgrad_norm: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "the maximum magnitude of L2 norms to perform per-sample gradient clipping"
        }
    )

    noise_scale: Optional[float] = field(
        default=-1.0,
        metadata={
            "help": "the privacy budget noise_scale"
        }
    )

    preclip: Optional[float] = field(
        default=0.0,
        metadata={"help": "preclip noise"}
    )

    neg_k: Optional[int] = field(
        default=8,
        metadata={"help": "number of negative samples"}
    )

    max_physical_batch_size: Optional[int] = field(
        default=12,
        metadata={"help": "max physcial batch size before splitting"}
    )

    dp_type: Optional[str] = field(
        default='edge',
        metadata={"help": "type of DP for relational data"}
    )

    use_peft: bool = field(default=False, metadata={"help": "use PEFT or not"})
    quantization: bool = field(default=False, metadata={"help": "use quantization or not"})
    resume_training: bool = field(default=False, metadata={"help": "Resume training from a checkpoint"})


@dataclass
class DenseEncodingArguments(TrainingArguments):
    use_gpu: bool = field(default=False, metadata={"help": "Use GPU for encoding"})
    encoded_save_path: str = field(default=None, metadata={"help": "where to save the encode"})
    save_path: str = field(default=None, metadata={"help": "where to save the result file"})
    retrieve_domain: str = field(default=None, metadata={"help": "name of the retrieve domain"})
    source_domain: str = field(default=None, metadata={"help": "name of the source domain"})
