#!/bin/bash

cd "$(dirname "$0")"
CURRENT_DIR="$(pwd)"
PROJ_DIR="$(dirname "$CURRENT_DIR")"

SDOMAIN=$2
BSIZE=$3
MODEL=$4

LOG_DIR=$PROJ_DIR/logs/$SDOMAIN/link_prediction
CHECKPOINT_DIR=$PROJ_DIR/data/ckpt/$SDOMAIN/link_prediction

if [[ "$SDOMAIN" = "sports" ]]
then
    TDOMAIN="cloth"
    RANK=8
elif [[ "$SDOMAIN" = "cloth" ]]
then
    TDOMAIN="sports"
    RANK=8
elif [[ "$SDOMAIN" = "mag_us" ]]
then
    TDOMAIN="mag_cn"
    RANK=4
elif [[ "$SDOMAIN" = "mag_cn" ]]
then
    TDOMAIN="mag_us"
    RANK=4
else
    echo "unsupported domain"
fi

PROCESSED_DIR=$PROJ_DIR/data/$TDOMAIN/link_prediction
EVAL_DIR=$PROJ_DIR/data/$TDOMAIN/link_prediction/valid.text.jsonl

RATIO=$(echo "scale=4; (16/$BSIZE) * 0.1" | bc)
STEPS=$((100/($BSIZE/16)))

MODEL_NAME='meta-llama/Llama-2-7b-hf'
TOKEN_NAME='meta-llama/Llama-2-7b-hf'
MODEL_TYPE=llama2
LR="1e-5"
NEGK=8

if [[ "$MODEL" = "large" ]]
then
    MODEL_NAME='bert-large-uncased'
    TOKEN_NAME='bert-large-uncased'
    MODEL_TYPE=bert_large
elif [[ "$MODEL" = "base" ]]
then
    MODEL_NAME='bert-base-uncased'
    TOKEN_NAME='bert-base-uncased'
    MODEL_TYPE=bert
elif [[ "$MODEL" = "sci" ]]
then
    MODEL_NAME="allenai/scibert_scivocab_uncased"
    TOKEN_NAME="allenai/scibert_scivocab_uncased"
    MODEL_TYPE=scibert
else
    echo "Using default model llama2-7b"
fi

echo "start training..."
echo $RATIO

CUDA_VISIBLE_DEVICES=$1 python Lora_SeqLP.py  \
    --output_dir $CHECKPOINT_DIR/$MODEL_TYPE/fewshot/base_$LR \
    --logging_dir $LOG_DIR/$MODEL_TYPE/$LR  \
    --model_name_or_sh $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --tokenizer_name $TOKEN_NAME \
    --config_name $TOKEN_NAME \
    --do_train  \
    --save_steps $STEPS  \
    --eval_steps $STEPS   \
    --logging_steps $(($STEPS/2)) \
    --train_path $PROCESSED_DIR/train.text.16_shot.jsonl  \
    --eval_path $EVAL_DIR  \
    --corpus_path $PROJ_DIR/data/$SDOMAIN/corpus.txt  \
    --num_train_epochs 500 \
    --per_device_train_batch_size $BSIZE  \
    --per_device_eval_batch_size 256 \
    --learning_rate $LR  \
    --warmup_ratio $RATIO \
    --max_len 32  \
    --save_total_limit 2 \
    --evaluation_strategy steps \
    --metric_for_best_model eval_mrr \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --report_to wandb \
    --seed 2024 \
    --set_pad_id \
    --fp16 \
    --use_peft \
    --lora_rank $RANK