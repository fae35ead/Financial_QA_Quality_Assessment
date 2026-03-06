import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)

# ================= 配置区 =================
INPUT_FILE = "../data/processed/labeled_golden_samples.csv"
OUTPUT_DIR = "../models/teacher_cross_encoder"
# 使用本地已下载的中文 RoBERTa-wwm-ext 模型目录，避免从 HuggingFace 在线下载
MODEL_NAME = "../models/pretrained_models/chinese-roberta-wwm-ext"

MAX_LEN = 384
BATCH_SIZE = 8
EPOCHS = 4
LEARNING_RATE = 2e-5


def compute_metrics(eval_pred):
    """定义评估指标：由于我们注重分类的均衡性，采用 Macro-F1 作为核心指标"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    # macro-f1 能够很好地评估数据分布均衡情况下的整体表现
    f1 = f1_score(labels, predictions, average='macro')
    return {"accuracy": acc, "f1_macro": f1}


def main():
    print("1. 正在加载并清理金标准数据...")
    df = pd.read_csv(INPUT_FILE)

    # 清理掉未能成功打标的数据（以防万一还有 -1 的遗留物）
    df = df[df['label'].isin([0, 1, 2])].copy()
    df['label'] = df['label'].astype(int)
    print(f"有效数据总量: {len(df)} 条。")

    # 划分训练集 (85%) 和验证集 (15%)
    # stratify=df['label'] 确保训练集和验证集的 0, 1, 2 比例完全一致
    train_df, val_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df['label'])
    print(f"训练集: {len(train_df)} 条, 验证集: {len(val_df)} 条。")

    # 转换为 HuggingFace Dataset 格式
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), MODEL_NAME))
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"本地模型目录不存在: {model_path}")

    print(f"\n2. 正在加载本地预训练分词器 (Tokenizer): {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    def tokenize_function(examples):
        """
        核心逻辑：构建 Cross-Encoder (交叉编码器) 格式
        当同时传入 text 和 text_pair 时，Tokenizer 会自动将它们拼成：
        [CLS] 提问 [SEP] 回答 [SEP]
        并在底层给它们分配不同的 segment_ids (0 和 1)
        """
        return tokenizer(
            text=examples["Qsubj"],
            text_pair=examples["Reply"],
            padding="max_length",
            truncation=True,
            max_length=MAX_LEN
        )

    print("3. 正在进行 Tokenize 编码 (将文本转化为张量)...")
    train_tokenized = train_dataset.map(tokenize_function, batched=True)
    val_tokenized = val_dataset.map(tokenize_function, batched=True)

    print(f"\n4. 正在初始化本地预训练模型 (3 分类器): {model_path}...")
    # num_labels=3 告诉模型我们在做三分类任务
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=3,
        local_files_only=True,
    )

    print("\n5. 配置训练参数 ...")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",  # 每个 epoch 结束时在验证集上评估一次
        save_strategy="epoch",  # 每个 epoch 保存一次检查点
        learning_rate=LEARNING_RATE,  # 微调的经典学习率
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,  # 推理时不计算梯度，显存占用少，可以翻倍
        num_train_epochs=EPOCHS,
        weight_decay=0.01,  # 防止过拟合
        fp16=True,  # 【关键】开启半精度加速，省显存，提速度！
        load_best_model_at_end=True,  # 训练结束后自动加载验证集 F1 最高的模型
        metric_for_best_model="f1_macro",  # 以 F1 作为挑选最好模型的指标
        logging_dir='./logs',  # TensorBoard 日志目录
        logging_steps=10,  # 每隔10步打印一次 loss
        report_to="none"  # 不使用 wandb 等外部记录工具
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        compute_metrics=compute_metrics
    )

    print("\n🚀 开始深度学习微调！启动你的 GPU 吧！")
    trainer.train()

    print("\n6. 正在保存微调后的最优 Teacher 模型...")
    # 保存最优模型和分词器，方便下个阶段直接调用
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ 模型已成功保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    # 如果出现显存碎片导致 OOM，可以取消下面这行的注释
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    main()