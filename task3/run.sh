#!/bin/bash

# Define lists for models and datasets
models=("roberta-base")
datasets=("restaurant_sup" "acl_sup")

# Set fixed random seed and training parameters
SEED=42
EPOCHS=4
BATCH_SIZE=64
LEARNING_RATE=1e-5

# Iterate over models and datasets
for MODEL_NAME in "${models[@]}"
do
    for DATASET_NAME in "${datasets[@]}"
    do

        # Run training 5 times
        for RUN in {1..5}
        do
            echo "Running model $MODEL_NAME on dataset $DATASET_NAME, iteration $RUN..."
            
            # Set output and logging directories to avoid overwriting
            OUTPUT_DIR="./output/${MODEL_NAME}_${DATASET_NAME}/run_$RUN"
            LOGGING_DIR="./logs/${MODEL_NAME}_${DATASET_NAME}/run_$RUN"
            mkdir -p $OUTPUT_DIR
            mkdir -p $LOGGING_DIR

            # Export RUN environment variable for wandb (optional)
            export RUN=$RUN

            # Run training script
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
