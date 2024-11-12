import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import evaluate
import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    TrainerCallback,
)
import wandb  # Import wandb for logging

from dataHelper import get_dataset

# Define model and data arguments
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Name or path of the pretrained model"}
    )
    
@dataclass
class DataTrainingArguments:
    dataset_name: str = field(
        metadata={"help": "Dataset name, or a comma-separated list of datasets"}
    )
    sep_token: str = field(
        default="<sep>", metadata={"help": "Separator token"}
    )
    max_seq_length: int = field(
        default=128, metadata={"help": "Maximum length of the input sequence"}
    )

def main():
    '''
    Initialize logging, random seed, argument parsing, etc.
    '''

    # Configure logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    # Parse arguments using HfArgumentParser
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    '''
    Initialize wandb for logging
    '''
    training_args.report_to = ["wandb"]
    training_args.log_level = "info"
    if 'wandb' in training_args.report_to:
        wandb.init(project="NLPDL", name=training_args.run_name)

    # Set random seed for reproducibility
    set_seed(training_args.seed)

    # Log training arguments
    logger.info(f"Training arguments: {training_args}")

    '''
    Load datasets
    '''
    # Load dataset(s) from dataHelper.py
    if ',' in data_args.dataset_name:
        dataset_names = [name.strip() for name in data_args.dataset_name.split(',')]
    else:
        dataset_names = [data_args.dataset_name]
    raw_datasets = get_dataset(dataset_names, sep_token="[SEP]")
    label_num = len(set(raw_datasets['train']['label']))

    '''
    Load model and tokenizer
    '''
    # Automatically load model configuration, tokenizer, and model
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=label_num)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)

    '''
    Process the dataset and create data collator
    '''
    def preprocess_function(examples):
        # Encode the text and add labels to the result
        result = tokenizer(
            examples['text'],
            padding=False,
            truncation=True,
            max_length=data_args.max_seq_length
        )
        result['label'] = examples['label']
        return result

    # Get the columns to remove after processing
    column_names = raw_datasets["train"].column_names
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        desc="Dataset preprocessing"
    )

    # Use DataCollatorWithPadding to handle batch padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    '''
    Define evaluation metrics
    '''
    # Load evaluation metrics
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(p):
        # Calculate accuracy and F1 scores
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = preds.argmax(-1)
        labels = p.label_ids
        acc = accuracy.compute(predictions=preds, references=labels)
        f1_micro = f1.compute(predictions=preds, references=labels, average='micro')
        f1_macro = f1.compute(predictions=preds, references=labels, average='macro')
        return {
            'accuracy': acc['accuracy'],
            'f1_micro': f1_micro['f1'],
            'f1_macro': f1_macro['f1']
        }

    '''
    Initialize Trainer
    '''
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_datasets['train'],
        eval_dataset=processed_datasets['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    '''
    Start training
    '''
    if training_args.do_train:
        logger.info("*** Starting training ***")
        trainer.train()

    '''
    Start evaluation
    '''
    if training_args.do_eval:
        logger.info("*** Starting evaluation ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)

    '''
    Finish wandb run
    '''
    if 'wandb' in training_args.report_to:
        wandb.finish()

if __name__ == "__main__":
    main()
