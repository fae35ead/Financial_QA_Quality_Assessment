import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ==========================================
# 0. 动态获取绝对路径
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 之前切分好的数据集路径
train_path = os.path.join(project_root, 'data', 'processed', 'train_dataset.csv')
valid_path = os.path.join(project_root, 'data', 'processed', 'valid_dataset.csv')
test_path = os.path.join(project_root, 'data', 'processed', 'test_dataset.csv')

# 模型检查点和最终模型的保存路径
checkpoint_dir = os.path.join(project_root, 'models', 'bert_checkpoints')
final_model_dir = os.path.join(project_root, 'models', 'final_bert_qa_model')

# ==========================================
# 1. 初始化模型基座与分词器
# ==========================================
MODEL_NAME = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ==========================================
# 2. 数据集加载与处理函数
# ==========================================
def load_hf_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到数据文件：{file_path}，请确认是否已运行切分脚本。")
    df = pd.read_csv(file_path)
    df = df[['model_input', 'final_label']]
    df = df.rename(columns={'final_label': 'label'})
    return Dataset.from_pandas(df)

def tokenize_function(examples):
    return tokenizer(
        examples["model_input"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# ==========================================
# 3. 主执行模块 (包含数据管道打通与模型训练)
# ==========================================
if __name__ == '__main__':
    # --- A. 数据管道处理 ---
    print("正在加载数据集...")
    train_dataset = load_hf_dataset(train_path)
    valid_dataset = load_hf_dataset(valid_path)
    test_dataset = load_hf_dataset(test_path)
    print(f"训练集规模: {len(train_dataset)} 条")

    # 小样本调试模式开关 (Dry Run)
    DEBUG_MODE = True  # 跑全量数据时，将这里改为 False
    DEBUG_SAMPLE_SIZE = 100000  # 假设只抽 100000 条训练集用来验证模型能否跑通

    if DEBUG_MODE:
        print(f"\n[⚠️ 调试模式已开启] 正在截取极小部分数据用于快速验证...")
        # 1.截取训练集 (取设定大小与实际长度的较小值，防越界)
        train_dataset = train_dataset.select(range(min(DEBUG_SAMPLE_SIZE, len(train_dataset))))

        # 2.验证集和测试集也按比例缩小 (比如取训练集的十分之一)
        eval_size = max(1, DEBUG_SAMPLE_SIZE // 10)
        valid_dataset = valid_dataset.select(range(min(eval_size, len(valid_dataset))))
        test_dataset = test_dataset.select(range(min(eval_size, len(test_dataset))))

        print(f"-> 缩小后规模 | 训练集: {len(train_dataset)} | 验证集: {len(valid_dataset)} | 测试集: {len(test_dataset)}\n")

    print("正在对数据进行 Tokenize 处理 (这可能需要几十秒)...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_valid = valid_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)

    columns_to_remove = ["model_input"]
    tokenized_train = tokenized_train.remove_columns(columns_to_remove)
    tokenized_valid = tokenized_valid.remove_columns(columns_to_remove)
    tokenized_test = tokenized_test.remove_columns(columns_to_remove)

    tokenized_train.set_format("torch")
    tokenized_valid.set_format("torch")
    tokenized_test.set_format("torch")
    print("✅ 数据管道打通！准备喂给模型。")

    # --- B. 硬件检测与模型加载 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🔥 当前炼丹炉已连接至: {device.type.upper()} !!!")
    if device.type != 'cuda':
        print("⚠️ 警告：没有检测到 GPU！训练可能极其缓慢。")

    print("正在下载并加载 BERT 模型结构...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    # --- C. 核心调参区 (专为本地 GPU 优化) ---
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,      # 减小单次输入，防止 OOM
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,      # 梯度累加，等效 Batch Size = 16
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        save_total_limit=2,
        fp16=True,                          # 开启半精度混合训练
    )

    # --- D. 组装 Trainer 并启动 ---
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        compute_metrics=compute_metrics,
    )

    print("\n🚀 点火！开始训练 QA 质量评估模型...")
    trainer.train()

    # --- E. 验收与持久化 ---
    print("\n🏆 训练完成！正在测试集上进行最终评估...")
    test_results = trainer.evaluate(tokenized_test)
    print(f"最终测试集准确率 (Accuracy): {test_results['eval_accuracy']:.4f}")

    print(f"\n正在将最佳模型和词表导出至：{final_model_dir}")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print("✅ 模型持久化保存完成。随时可以调用推理！")