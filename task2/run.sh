#!/bin/bash

# 定义模型和数据集列表
models=("roberta-base" "bert-base-uncased" "allenai/scibert_scivocab_uncased")
datasets=("agnews_sup")
# "restaurant_sup" "acl_sup" 
# 固定随机种子
SEED=42
EPOCHS=4
BATCH_SIZE=64
LEARNING_RATE=1e-5

# 遍历模型和数据集
for MODEL_NAME in "${models[@]}"
do
    for DATASET_NAME in "${datasets[@]}"
    do

        # 运行 5 次训练
        for RUN in {1..5}
        do
            echo "运行模型 $MODEL_NAME 在数据集 $DATASET_NAME 上的第 $RUN 次训练..."
            # 设置输出和日志目录，避免覆盖
            OUTPUT_DIR="./output/${MODEL_NAME}_${DATASET_NAME}/run_$RUN"
            LOGGING_DIR="./logs/${MODEL_NAME}_${DATASET_NAME}/run_$RUN"
            mkdir -p $OUTPUT_DIR
            mkdir -p $LOGGING_DIR

            # 导出 RUN 环境变量，供 wandb 使用（可选）
            export RUN=$RUN

            # 运行训练脚本
            python train.py \
                --model_name_or_path $MODEL_NAME \
                --dataset_name $DATASET_NAME \
                --do_train \
                --do_eval \
                --evaluation_strategy epoch \
                --logging_strategy steps \
                --logging_steps 10 \
                --per_device_train_batch_size $BATCH_SIZE \
                --per_device_eval_batch_size $BATCH_SIZE \
                --learning_rate $LEARNING_RATE \
                --num_train_epochs $EPOCHS \
                --output_dir $OUTPUT_DIR \
                --logging_dir $LOGGING_DIR \
                --report_to wandb \
                --seed $SEED
        done
    done
done
