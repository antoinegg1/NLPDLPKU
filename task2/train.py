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
import wandb  # 导入 wandb

from dataHelper import get_dataset

# 定义数据和模型的参数
@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "预训练模型的名称或路径"}
    )
    
@dataclass
class DataTrainingArguments:
    dataset_name: str = field(
        metadata={"help": "数据集的名称，或者列表，逗号分隔"}
    )
    sep_token: str = field(
        default="<sep>", metadata={"help": "分隔符 token"}
    )
    max_seq_length: int = field(
        default=128, metadata={"help": "输入序列的最大长度"}
    )
def main():
    '''
    初始化 logging、seed、argparse...
    '''

    # 配置 logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    # 使用 HfArgumentParser 解析参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    '''
        初始化 wandb
    '''
    training_args.report_to = ["wandb"]
    training_args.log_level = "info"
    if 'wandb' in training_args.report_to:
        wandb.init(project="NLPDL", name=training_args.run_name)
    # 设置随机种子
    set_seed(training_args.seed)

    # 日志参数
    logger.info(f"训练参数：{training_args}")


    

    '''
        加载数据集
    '''
    # 从 dataHelper.py 中加载数据集
    if ',' in data_args.dataset_name:
        dataset_names = [name.strip() for name in data_args.dataset_name.split(',')]
    else:
        dataset_names = [data_args.dataset_name]
    raw_datasets = get_dataset(dataset_names, sep_token="[SEP]")
    label_num=len(set(raw_datasets['train']['label']))

    '''
        加载模型和 tokenizer
    '''
    # 自动加载模型配置、tokenizer 和模型
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, num_labels=label_num)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)

    '''
        处理数据集并构建数据 collator
    '''
    def preprocess_function(examples):
        # 对文本进行编码
        result = tokenizer(
            examples['text'],
            padding=False,
            truncation=True,
            max_length=data_args.max_seq_length
        )
        # 将 labels 添加到结果中
        result['label'] = examples['label']
        return result

    # 获取需要删除的列
    column_names = raw_datasets["train"].column_names
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=column_names,
        desc="数据集预处理"
    )

    # 使用 DataCollatorWithPadding 处理批量数据
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    '''
        定义评估指标
    '''
    # 加载评估指标
    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    def compute_metrics(p):
        preds = p.predictions.argmax(-1)
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
        初始化 Trainer
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
        开始训练
    '''
    if training_args.do_train:
        logger.info("*** 开始训练 ***")
        trainer.train()

    '''
        开始评估
    '''
    if training_args.do_eval:
        logger.info("*** 开始评估 ***")
        metrics = trainer.evaluate()

        trainer.log_metrics("eval", metrics)

    '''
        训练完成，关闭 wandb 运行
    '''
    if 'wandb' in training_args.report_to:
        wandb.finish()

if __name__ == "__main__":
    main()
