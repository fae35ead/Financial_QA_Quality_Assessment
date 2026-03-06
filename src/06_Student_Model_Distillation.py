import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# ================= 配置区 =================
INPUT_FILE = "../data/processed/soft_labeled_20k.csv"
OUTPUT_DIR = "../models/student_distilbert"

MODEL_NAME = r"F:\软件\学习相关\PycharmProjects\QA\models\pretrained_models\distilbert-base-zh-cased"

MAX_LEN = 384
BATCH_SIZE = 16  # Student 比较小，显存占用低，Batch Size 可以开大
EPOCHS = 4
LEARNING_RATE = 3e-5


# ================= 自定义蒸馏 Trainer =================
class DistillationTrainer(Trainer):
    """
    重写 Hugging Face 的 Trainer，使其支持软标签（Soft Labels）的 KL 散度损失计算。
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1. 必须用 pop 把自定义的软标签特征拿出来
        # 否则这些非标准参数喂给底层 forward 函数会报 "unexpected keyword argument" 错误
        teacher_probs = inputs.pop("teacher_probs")

        # 2. 拿到真实硬标签（但把它 pop 掉，不要让模型底层去算自带的交叉熵，避免冲突）
        if "labels" in inputs:
            labels = inputs.pop("labels")

        # 3. 让 Student 模型进行前向传播
        outputs = model(**inputs)
        student_logits = outputs.logits

        # 4. 【核心魔法：计算 KL 散度】
        # 将 Student 的 Logits 转化为 Log-Probabilities (对数概率)
        student_log_probs = F.log_softmax(student_logits, dim=-1)

        # 使用 KLDivLoss 计算 Student 预测分布与 Teacher 目标分布的差异
        # reduction="batchmean" 是 PyTorch 官方推荐的计算 KL 散度的标准操作
        loss_fct = nn.KLDivLoss(reduction="batchmean")
        loss = loss_fct(student_log_probs, teacher_probs)

        return (loss, outputs) if return_outputs else loss


def main():
    print("1. 正在加载带软标签的 2 万条蒸馏数据...")
    df = pd.read_csv(INPUT_FILE)

    # 划分训练集 (90%) 和验证集 (10%)
    train_df, val_df = train_test_split(df, test_size=0.10, random_state=42)
    print(f"训练集: {len(train_df)} 条, 验证集: {len(val_df)} 条。")

    # 转换为 HuggingFace Dataset 格式
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    print(f"\n2. 正在加载本地 Student 模型分词器...")
    # 从本地路径加载
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        # 对文本进行交叉编码器格式的拼接 [CLS] Q [SEP] R [SEP]
        tokenized = tokenizer(
            text=examples["Qsubj"],
            text_pair=examples["Reply"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN
        )
        # 将我们上一步生成的 prob_0, prob_1, prob_2 打包成一个张量格式供 Loss 函数调用
        tokenized["teacher_probs"] = [
            [p0, p1, p2] for p0, p1, p2 in zip(examples["prob_0"], examples["prob_1"], examples["prob_2"])
        ]
        # 保留硬标签，以便在非训练阶段（比如 eval 评估时）进行准确率计算
        tokenized["labels"] = examples["teacher_hard_label"]
        return tokenized

    print("3. 正在进行 Tokenize 编码与特征打包...")

    # 获取原始数据集的所有列名（比如 Question, Reply, From, prob_0 等）
    original_columns = train_dataset.column_names

    # 在分词后，利用 remove_columns 把原始的字符串列全部扔掉！
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=original_columns
    )
    val_tokenized = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=original_columns
    )

    print(f"\n4. 正在初始化极速 Student 模型 (3 分类器)...")
    # 从本地路径加载基座权重
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    print("\n5. 配置蒸馏训练参数 (针对 RTX 2070S 优化)...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        fp16=True,  # 开启半精度，压榨 RTX 2070S 性能
        logging_dir='./logs_distil',
        logging_steps=50,
        report_to="none",
        remove_unused_columns=False  # 【极其重要】不能删掉 teacher_probs！
    )

    # 实例化我们自定义的软标签蒸馏 Trainer
    trainer = DistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
    )

    print("\n🚀 开始终极蒸馏！让本地 DistilBERT 吸收 Teacher 的暗知识吧！")
    trainer.train()

    print("\n6. 正在保存满级出山的 Student 模型...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Student 模型已成功保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()