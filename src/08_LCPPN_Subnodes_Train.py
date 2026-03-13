'''该文件负责 LCPPN 的子节点分类器训练，包含 Direct-5 分类财务节点和 Evasive-4 分类战术节点两个分支。核心亮点在于：
1. 数据增强隔离：合成数据只混入训练集，验证集保持纯洁的真实数据，确保模型评估的真实性和可靠性。
2. Focal Loss 定制 Trainer：针对类别不平衡问题，定制了 Focal Loss 版本的 Trainer，提升模型对少数类的学习能力。
3. 训练结束即考试：训练完成后，立即在纯真实验证集上进行评测，输出详细的分类报告，确保模型的实战表现符合预期。
'''

import os
from pathlib import Path
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
PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_DATA_FILE = PROJECT_ROOT / "data" / "processed" / "labeled_subnodes_samples.csv"
SYNTHETIC_DATA_FILE = PROJECT_ROOT / "data" / "processed" / "synthetic_augmented_samples.csv"

OUTPUT_DIR_DIRECT = PROJECT_ROOT / "models" / "lcppn_direct_classifier"
OUTPUT_DIR_EVASIVE = PROJECT_ROOT / "models" / "lcppn_evasive_classifier"
MODEL_NAME = PROJECT_ROOT / "models" / "pretrained_models" / "distilbert-base-zh-cased"

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


def safe_split_real_data(df_real_node, node_name, test_size=0.15, random_state=42):
    """只基于真实样本切分训练/验证集，避免切分前复制导致的数据泄露。"""
    df_real_node = df_real_node.copy()
    label_counts = df_real_node['label_id'].value_counts().sort_index()

    if len(df_real_node) < 2:
        raise ValueError(f"{node_name} 真实样本数不足 2 条，无法切分训练/验证集。")

    # 分层切分要求：每个类别至少 2 条，且验证集容量不能小于类别数
    can_stratify = (
        label_counts.min() >= 2 and
        max(1, int(np.ceil(len(df_real_node) * test_size))) >= label_counts.shape[0]
    )

    if can_stratify:
        train_real, val_df = train_test_split(
            df_real_node,
            test_size=test_size,
            random_state=random_state,
            stratify=df_real_node['label_id']
        )
        print("✅ 真实数据采用分层切分，验证集类别分布更稳定。")
        return train_real.reset_index(drop=True), val_df.reset_index(drop=True)

    print("⚠️ 真实数据过少或存在极小类，无法安全分层切分；改为非分层切分以避免数据泄露。")
    train_real, val_df = train_test_split(
        df_real_node,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )
    return train_real.reset_index(drop=True), val_df.reset_index(drop=True)


def train_sub_node(df_real_node, df_syn_node, labels_dict, output_dir, node_name):
    print(f"\n{'=' * 50}")
    print(f"🚀 启动 LCPPN {node_name} 训练 (含数据增强隔离)")
    print(f"{'=' * 50}")

    df_real_node = df_real_node.copy()
    df_real_node['label_id'] = df_real_node['sub_label'].map(labels_dict)
    df_real_node = df_real_node.dropna(subset=['label_id']).copy()
    df_real_node['label_id'] = df_real_node['label_id'].astype(int)

    if df_real_node.empty:
        raise ValueError(f"{node_name} 没有可用于训练的真实样本。")

    print("📌 真实样本类别分布：")
    print(df_real_node['sub_label'].value_counts())

    # 【核心逻辑 1】：先只切分真实数据，避免切分前复制产生数据泄露
    train_real, val_df = safe_split_real_data(df_real_node, node_name=node_name, test_size=0.15, random_state=42)

    # 【核心逻辑 2】：合成数据只进入训练集，绝不进入验证集
    syn_count_used = 0
    syn_count_dropped = 0
    if df_syn_node is not None and not df_syn_node.empty:
        df_syn_node = df_syn_node.copy()
        df_syn_node['label_id'] = df_syn_node['sub_label'].map(labels_dict)
        syn_count_dropped = int(df_syn_node['label_id'].isna().sum())
        df_syn_node = df_syn_node.dropna(subset=['label_id']).copy()
        df_syn_node['label_id'] = df_syn_node['label_id'].astype(int)
        syn_count_used = len(df_syn_node)
        train_df = pd.concat([train_real, df_syn_node], ignore_index=True)
        print(
            f"🔥 数据增强启动！真实训练集 {len(train_real)} 条 + 合成数据 {syn_count_used} 条 -> 扩充后训练集 {len(train_df)} 条"
        )
        if syn_count_dropped > 0:
            print(f"⚠️ 有 {syn_count_dropped} 条合成样本因 sub_label 不在当前节点标签表中而被跳过。")
    else:
        train_df = train_real.copy()
        print(f"无需增强，训练集共 {len(train_df)} 条")

    print(f"🛡️ 纯净验证集共 {len(val_df)} 条 (绝对不含合成假数据)")

    # 重新计算分布和 Focal Loss 的对数平滑权重 (基于扩充后的 train_df)
    class_counts = train_df['label_id'].value_counts().reindex(range(len(labels_dict)), fill_value=0)
    valid_class_counts = class_counts[class_counts > 0].values
    total_samples = int(valid_class_counts.sum())
    weights = np.ones(len(labels_dict), dtype=np.float32)
    if total_samples > 0 and len(valid_class_counts) > 0:
        for class_id, count in class_counts.items():
            if count > 0:
                weights[class_id] = np.log(total_samples / count)
    weights = torch.tensor(weights, dtype=torch.float)

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_NAME))

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

    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_NAME), num_labels=len(labels_dict))

    use_fp16 = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        fp16=use_fp16,
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

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
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
