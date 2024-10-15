#!/bin/bash

cd "$(dirname "$0")"
PROJ_DIR="$(pwd)"

# Parse input parameters
SDOMAIN=$2
BSIZE=$3
MODEL=$4

if [[ "$SDOMAIN" = "sports" ]]
then
    TDOMAIN="cloth"
elif [[ "$SDOMAIN" = "cloth" ]]
then
    TDOMAIN="sports"
elif [[ "$SDOMAIN" = "mag_us" ]]
then
    TDOMAIN="mag_cn"
elif [[ "$SDOMAIN" = "mag_cn" ]]
then
    TDOMAIN="mag_us"
else
    echo "unsupported domain"
fi

PROCESSED_DIR=$PROJ_DIR/data/$SDOMAIN/link_prediction
LOG_DIR=$PROJ_DIR/logs/$SDOMAIN/link_prediction
CHECKPOINT_DIR=$PROJ_DIR/data/ckpt/$SDOMAIN/link_prediction
EVAL_DIR=$PROJ_DIR/data/$TDOMAIN/link_prediction/valid.text.jsonl

RATIO=$(echo "scale=4; (16/$BSIZE) * 0.1" | bc)
STEPS=$((1000/($BSIZE/32)))
LOG_STEPS=$(($STEPS/2))

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
    template_file="template/bert_train_config.json"
elif [[ "$MODEL" = "base" ]]
then
    MODEL_NAME='bert-base-uncased'
    TOKEN_NAME='bert-base-uncased'
    MODEL_TYPE=bert
    template_file="template/bert_train_config.json"
elif [[ "$MODEL" = "sci" ]]
then
    MODEL_NAME="allenai/scibert_scivocab_uncased"
    TOKEN_NAME="allenai/scibert_scivocab_uncased"
    MODEL_TYPE=scibert
    template_file="template/bert_train_config.json"
else
    MODEL="llama2-7b"
    echo "Using default model llama2-7b-hf"
    template_file="template/llama_train_config.json"
fi

echo "start training..."
echo $RATIO

# Load JSON template
if [ ! -f "$template_file" ]; then
  echo "Error: Template file '$template_file' not found."
  exit 1
fi

# Modify JSON template
json_data=$(jq --arg output_dir $CHECKPOINT_DIR/$MODEL_TYPE/nondp/$LR'_b'$BSIZE'_k'$NEGK \
               --arg logging_dir $LOG_DIR/$MODEL_TYPE/$LR \
               --arg dataset_name $SDOMAIN \
               --arg train_path $PROCESSED_DIR/train.text.new.jsonl \
               --arg corpus_path $PROJ_DIR/data/$SDOMAIN/corpus.txt \
               --arg MODEL_NAME $MODEL_NAME \
               --arg MODEL_TYPE $MODEL_TYPE \
               --arg TOKEN_NAME $TOKEN_NAME \
               --arg EVAL_DIR $EVAL_DIR \
               --argjson STEPS $STEPS \
               --argjson LOG_STEPS $LOG_STEPS \
               --argjson BSIZE $BSIZE \
               --argjson LR $LR \
               --argjson RATIO $RATIO \
               --argjson NEGK $NEGK \
               '.pdebug=false | .output_dir=$output_dir | .logging_dir=$logging_dir | .model_name_or_path=$MODEL_NAME | .model_type=$MODEL_TYPE | .tokenizer_name=$TOKEN_NAME | .config_name=$TOKEN_NAME | .save_steps=$STEPS | .eval_steps=$STEPS | .logging_steps=$LOG_STEPS | .dataset_name=$dataset_name | .train_path=$train_path | .eval_path=$EVAL_DIR | .corpus_path=$corpus_path | .per_device_train_batch_size=$BSIZE | .learning_rate=$LR | .warmup_ratio=$RATIO' "$template_file")

# Save modified JSON to a new file
timestamp=$(date +'%Y%m%d_%H%M%S')
json_file="logs/config_${MODEL}_${timestamp}.json"
echo "$json_data" > "$json_file"
echo "Training configs are saved to $json_file"

CUDA_VISIBLE_DEVICES=$1 python Lora_SeqLP_prv.py $json_file