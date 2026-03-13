'''该文件实现了一个完整的蒸馏训练流程，使用 Hugging Face 的 Transformers 库来训练一个 Student 模型（DistilBERT），以模仿一个 Teacher 模型（如 BERT）的输出分布。
核心在于自定义 Trainer 来计算 KL 散度损失，使 Student 学习 Teacher 的软标签知识。整个流程包括数据加载、划分、编码、模型初始化、训练和保存。

补充说明：
1. 蒸馏训练通常会引入温度 T。T>1 会把类别分布拉平，让 Student 更容易学习 Teacher 的“暗知识”（非最大类之间的相对关系）。
2. 推理阶段通常不需要继续使用高温度，默认按 T=1 直接用原始 logits 做分类即可。
3. 针对标签 1（避重就轻）样本偏少、边界模糊的问题，这里额外混入一个“硬标签监督项”，并支持 Class Weights / Focal Loss。
'''

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# ================= 配置区 =================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "soft_labeled_100k.csv"
OUTPUT_DIR = PROJECT_ROOT / "models" / "student_100k_T4_plus_FL_distilbert"

MODEL_NAME = PROJECT_ROOT / "models" / "pretrained_models" / "distilbert-base-zh-cased"

MAX_LEN = 384
BATCH_SIZE = 16  # Student 比较小，显存占用低，Batch Size 可以开大
EPOCHS = 4
LEARNING_RATE = 3e-5
DISTILL_TEMPERATURE = 4.0  # 建议从 2/4/8 做对比实验；推理阶段仍按 T=1 使用模型输出
SOFT_LOSS_WEIGHT = 0.7     # 软标签蒸馏损失权重
HARD_LOSS_WEIGHT = 0.3     # 硬标签监督损失权重；两者相加建议为 1
CLASS_WEIGHTS = [1.0, 1.5, 1.0]  # 对标签 1 增加惩罚力度，可尝试 1.5/1.8/2.0
USE_FOCAL_LOSS = True     # 先建议从加权交叉熵开始；效果不够再改为 True
FOCAL_GAMMA = 2.0


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average='macro')
    precision, recall, f1_each, _ = precision_recall_fscore_support(
        labels,
        preds,
        labels=[0, 1, 2],
        zero_division=0
    )
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "class1_precision": precision[1],
        "class1_recall": recall[1],
        "class1_f1": f1_each[1],
    }


# ================= 自定义蒸馏 Trainer =================
class DistillationTrainer(Trainer):
    """
    重写 Hugging Face 的 Trainer，使其支持软标签蒸馏 + 硬标签加权监督。
    """

    def __init__(self, *args, temperature=1.0, soft_loss_weight=0.7,
                 hard_loss_weight=0.3, class_weights=None,
                 use_focal_loss=False, focal_gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.soft_loss_weight = soft_loss_weight
        self.hard_loss_weight = hard_loss_weight
        self.class_weights = class_weights
        self.use_focal_loss = use_focal_loss
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        teacher_probs = inputs.pop("teacher_probs")
        labels = inputs.pop("labels") if "labels" in inputs else None

        outputs = model(**inputs)
        student_logits = outputs.logits
        device = student_logits.device

        # 1) 软标签蒸馏损失：保留 Teacher 的分布知识
        temperature = self.temperature
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = teacher_probs.to(device).float()
        soft_loss_fct = nn.KLDivLoss(reduction="batchmean")
        soft_loss = soft_loss_fct(student_log_probs, teacher_probs) * (temperature ** 2)

        # 2) 硬标签监督损失：重点照顾标签 1 这类少样本、边界模糊类别
        hard_loss = torch.tensor(0.0, device=device)
        if labels is not None:
            labels = labels.to(device).long()
            if self.class_weights is not None:
                class_weights = self.class_weights.to(device)
            else:
                class_weights = None

            if self.use_focal_loss:
                ce_loss = F.cross_entropy(
                    student_logits,
                    labels,
                    reduction='none',
                    weight=class_weights
                )
                pt = torch.exp(-ce_loss)
                hard_loss = ((1 - pt) ** self.focal_gamma * ce_loss).mean()
            else:
                hard_loss = F.cross_entropy(student_logits, labels, weight=class_weights)

        # 3) 混合总损失：默认以蒸馏为主，硬标签纠偏为辅
        loss = self.soft_loss_weight * soft_loss + self.hard_loss_weight * hard_loss
        return (loss, outputs) if return_outputs else loss


def main():
    print("1. 正在加载带软标签的 10 万条蒸馏数据...")
    print(f"   当前蒸馏温度 T = {DISTILL_TEMPERATURE}（仅训练时生效，推理默认仍为 T=1）")
    print(f"   当前损失配比: soft={SOFT_LOSS_WEIGHT}, hard={HARD_LOSS_WEIGHT}")
    print(f"   当前类别权重: {CLASS_WEIGHTS} | Focal Loss: {'开启' if USE_FOCAL_LOSS else '关闭'}")
    df = pd.read_csv(INPUT_FILE)

    # 划分训练集 (90%) 和验证集 (10%)
    stratify_col = df['teacher_hard_label'] if 'teacher_hard_label' in df.columns else None
    train_df, val_df = train_test_split(df, test_size=0.10, random_state=42, stratify=stratify_col)
    print(f"训练集: {len(train_df)} 条, 验证集: {len(val_df)} 条。")

    # 转换为 HuggingFace Dataset 格式
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    print(f"\n2. 正在加载本地 Student 模型分词器...")
    # 从本地路径加载
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_NAME))

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
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_NAME), num_labels=3)

    print("\n5. 配置蒸馏训练参数 (针对 RTX 2070S 优化)...")
    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        fp16=use_fp16,
        logging_dir=str((PROJECT_ROOT / "logs_distil").resolve()),
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
        temperature=DISTILL_TEMPERATURE,
        soft_loss_weight=SOFT_LOSS_WEIGHT,
        hard_loss_weight=HARD_LOSS_WEIGHT,
        class_weights=torch.tensor(CLASS_WEIGHTS, dtype=torch.float),
        use_focal_loss=USE_FOCAL_LOSS,
        focal_gamma=FOCAL_GAMMA,
        compute_metrics=compute_metrics,
    )

    print("\n🚀 开始终极蒸馏！让本地 DistilBERT 吸收 Teacher 的暗知识吧！")
    trainer.train()

    print("\n6. 正在保存满级出山的 Student 模型...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"✅ Student 模型已成功保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
