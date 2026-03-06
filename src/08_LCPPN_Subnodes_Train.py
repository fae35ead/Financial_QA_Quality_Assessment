# 该文件负责训练 LCPPN 的两个核心子节点分类器：Direct-5 分类财务节点和 Evasive-4 分类战术节点。
# 训练过程中，我们将真实数据和合成数据进行隔离处理：真实数据仅用于构建纯洁的验证集，而合成数据则全部混入训练集以增强模型的泛化能力。

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# ================= 配置区 =================
REAL_DATA_FILE = "../data/processed/labeled_subnodes_samples.csv"
SYNTHETIC_DATA_FILE = "../data/processed/synthetic_augmented_samples.csv"

# 绝对路径黑科技（防 FileNotFoundError）
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))

OUTPUT_DIR_DIRECT = os.path.join(project_root, "models", "lcppn_direct_classifier")
OUTPUT_DIR_EVASIVE = os.path.join(project_root, "models", "lcppn_evasive_classifier")
MODEL_NAME = r"F:\软件\学习相关\PycharmProjects\QA\models\pretrained_models\distilbert-base-zh-cased"

MAX_LEN = 384
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 2e-5

DIRECT_LABELS = {
    "资本运作与并购": 0, "技术与研发进展": 1,
    "产能与项目规划": 2, "合规与风险披露": 3, "财务表现指引": 4
}
EVASIVE_LABELS = {
    "推迟回答": 0, "转移话题": 1,
    "战略性模糊": 2, "外部归因": 3
}


# ================= 黑科技：Focal Loss 自定义 Trainer =================
class FocalLossTrainer(Trainer):
    def __init__(self, *args, class_weights=None, gamma=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None
        self.gamma = gamma

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        ce_loss = F.cross_entropy(logits, labels, reduction='none', weight=self.class_weights)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return (focal_loss, outputs) if return_outputs else focal_loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "f1_macro": f1}


def prepare_and_align_data():
    """读取真实数据和合成数据，只提取模型需要的核心三列，并对齐列名"""
    df_real = pd.read_csv(REAL_DATA_FILE)
    df_syn = pd.read_csv(SYNTHETIC_DATA_FILE)

    # 统一列名：如果叫 Qsubj，一律改为 Question
    if 'Qsubj' in df_real.columns: df_real.rename(columns={'Qsubj': 'Question'}, inplace=True)
    if 'Qsubj' in df_syn.columns: df_syn.rename(columns={'Qsubj': 'Question'}, inplace=True)

    # 提取需要的列，抛弃 reason、confidence 等无用列
    cols_to_keep = ['Question', 'Reply', 'label', 'sub_label']
    df_real = df_real[cols_to_keep].copy()
    df_syn = df_syn[cols_to_keep].copy()

    df_real['is_synthetic'] = False
    df_syn['is_synthetic'] = True

    return df_real, df_syn


def train_sub_node(df_real_node, df_syn_node, labels_dict, output_dir, node_name):
    print(f"\n{'=' * 50}")
    print(f"🚀 启动 LCPPN {node_name} 训练 (含数据增强隔离)")
    print(f"{'=' * 50}")

    df_real_node = df_real_node.copy()
    df_real_node['label_id'] = df_real_node['sub_label'].map(labels_dict)

    # 针对真实数据的极小样本进行保护，防止 sklearn stratify 崩溃
    for label_name, count in df_real_node['sub_label'].value_counts().items():
        if count < 4:
            rare_samples = df_real_node[df_real_node['sub_label'] == label_name]
            # 复制几份凑够切分的最低限度
            df_real_node = pd.concat([df_real_node, rare_samples, rare_samples], ignore_index=True)

    # 【核心逻辑 1】：真实数据切分 85% 训练，15% 验证（纯洁的验证集！）
    train_real, val_df = train_test_split(df_real_node, test_size=0.15, random_state=42,
                                          stratify=df_real_node['label_id'])

    # 【核心逻辑 2】：合成数据打上 label_id 后，全部混入训练集
    if df_syn_node is not None and not df_syn_node.empty:
        df_syn_node = df_syn_node.copy()
        df_syn_node['label_id'] = df_syn_node['sub_label'].map(labels_dict)
        train_df = pd.concat([train_real, df_syn_node], ignore_index=True)
        print(
            f"🔥 数据增强启动！真实训练集 {len(train_real)} 条 + 合成数据 {len(df_syn_node)} 条 -> 扩充后训练集 {len(train_df)} 条")
    else:
        train_df = train_real
        print(f"无需增强，训练集共 {len(train_df)} 条")

    print(f"🛡️ 纯净验证集共 {len(val_df)} 条 (绝对不含合成假数据)")

    # 重新计算分布和 Focal Loss 的对数平滑权重 (基于扩充后的 train_df)
    class_counts = train_df['label_id'].value_counts().sort_index().values
    total_samples = sum(class_counts)
    weights = np.log(total_samples / class_counts)
    weights = torch.tensor(weights, dtype=torch.float)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_func(examples):
        tokenized = tokenizer(
            text=examples["Question"],
            text_pair=examples["Reply"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN
        )
        tokenized["labels"] = examples["label_id"]
        return tokenized

    train_dataset = Dataset.from_pandas(train_df).map(tokenize_func, batched=True,
                                                      remove_columns=train_df.columns.tolist())
    val_dataset = Dataset.from_pandas(val_df).map(tokenize_func, batched=True, remove_columns=val_df.columns.tolist())

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(labels_dict))

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to="none"
    )

    trainer = FocalLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        class_weights=weights,
        gamma=2.0
    )

    trainer.train()

    # 训练结束后，立刻用这最纯洁的验证集考一次期末考试
    print(f"\n📊 【{node_name}】期末考试 (在纯真实验证集上的表现):")
    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids
    target_names = [k for k, v in sorted(labels_dict.items(), key=lambda item: item[1])]
    print(classification_report(labels, preds, target_names=target_names))

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ 【{node_name}】模型已保存！\n")


def main():
    df_real, df_syn = prepare_and_align_data()

    # 1. Direct 分支 (Root == 0)
    df_real_direct = df_real[df_real['label'] == 0]
    df_syn_direct = df_syn[df_syn['label'] == 0]
    train_sub_node(df_real_direct, df_syn_direct, DIRECT_LABELS, OUTPUT_DIR_DIRECT, "Direct-5分类财务节点")

    # 2. Evasive 分支 (Root == 2)
    df_real_evasive = df_real[df_real['label'] == 2]
    df_syn_evasive = df_syn[df_syn['label'] == 2]
    train_sub_node(df_real_evasive, df_syn_evasive, EVASIVE_LABELS, OUTPUT_DIR_EVASIVE, "Evasive-4分类战术节点")


if __name__ == "__main__":
    main()