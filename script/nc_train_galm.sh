cd "$(dirname "$0")"
CURRENT_DIR="$(pwd)"
PROJ_DIR="$(dirname "$CURRENT_DIR")"

SDOMAIN=$2
BSIZE=$3
MODEL=$4

CHECKPOINT_DIR=$PROJ_DIR/data/ckpt/$SDOMAIN/link_prediction

if [[ "$SDOMAIN" = "sports" ]]
then
    TDOMAIN="cloth"
    CNUM=9
elif [[ "$SDOMAIN" = "cloth" ]]
then
    TDOMAIN="sports"
    CNUM=16
elif [[ "$SDOMAIN" = "mag_us" ]]
then
    TDOMAIN="mag_cn"
    CNUM=40
elif [[ "$SDOMAIN" = "mag_cn" ]]
then
    TDOMAIN="mag_us"
    CNUM=40
else
    echo "unsupported domain"
fi

PROCESSED_DIR=$PROJ_DIR/data/$TDOMAIN/nc-coarse/8_8
LOG_DIR=$PROJ_DIR/logs/$SDOMAIN/nc_class
SAVE_DIR=$PROJ_DIR/data/ckpt/$SDOMAIN/nc_class

RATIO=0.1
STEPS=100

MODEL_NAME='meta-llama/Llama-2-7b-hf'
TOKEN_NAME='meta-llama/Llama-2-7b-hf'
MODEL_TYPE=llama2
LR="1e-5"
NEGK=8

echo "start training..."
echo $RATIO

CUDA_VISIBLE_DEVICES=$1 python Lora_SeqCLS.py  \
    --output_dir $SAVE_DIR/$MODEL_TYPE/base_$LR \
    --logging_dir $LOG_DIR/$MODEL_TYPE/$LR  \
    --model_name_or_path $MODEL_NAME \
    --model_type $MODEL_TYPE \
    --tokenizer_name $TOKEN_NAME \
    --config_name $TOKEN_NAME \
    --do_train  \
    --save_steps 1000 \
    --eval_steps $STEPS  \
    --logging_steps $(($STEPS/2)) \
    --train_path $PROCESSED_DIR/train.text.jsonl  \
    --eval_path $PROCESSED_DIR/val.text.jsonl  \
    --test_path $PROCESSED_DIR/test.text.jsonl  \
    --class_num $CNUM \
    --num_train_epochs 500 \
    --per_device_train_batch_size $BSIZE  \
    --per_device_eval_batch_size 256 \
    --learning_rate $LR  \
    --warmup_ratio $RATIO \
    --max_len 32  \
    --save_total_limit 2 \
    --evaluation_strategy steps \
    --metric_for_best_model eval_F1_macro \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --report_to wandb \
    --seed 2024 \
    --set_pad_id \
    --fp16 \
    --fix_bert