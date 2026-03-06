import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm

# ================= 配置区 =================
# 加载你最初用 LLM 打好标的 1000 条金标准数据
TEST_DATA_FILE = "../data/processed/labeled_golden_samples.csv"
# 加载我们刚刚新鲜出炉的 Student 模型
STUDENT_MODEL_DIR = "../models/student_distilbert"

MAX_LEN = 384
BATCH_SIZE = 32


class QATestDataset(Dataset):
    def __init__(self, texts_q, texts_r, labels, tokenizer, max_len):
        self.texts_q = texts_q
        self.texts_r = texts_r
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts_q)

    def __getitem__(self, item):
        q = str(self.texts_q[item])
        r = str(self.texts_r[item])
        label = int(self.labels[item])

        encoding = self.tokenizer(
            text=q,
            text_pair=r,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def main():
    print("1. 正在加载 1000 条金标准测试集...")
    df = pd.read_csv(TEST_DATA_FILE)

    # 清洗掉可能存在的 -1 失败数据
    df = df[df['label'].isin([0, 1, 2])].copy()
    print(f"有效测试数据: {len(df)} 条")

    questions = df['Qsubj'].tolist()
    replies = df['Reply'].tolist()
    labels = df['label'].tolist()

    print("\n2. 正在请出我们的满级 Student 模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(STUDENT_MODEL_DIR)
    model.to(device)
    model.eval()

    dataset = QATestDataset(questions, replies, labels, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_preds = []
    all_labels = []

    print("\n3. 🚀 开始极速闭卷考试...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Student 答题中"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            true_labels = batch['labels'].numpy()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # 获取概率最高的那一项作为最终预测标签
            preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(true_labels)

    print("\n4. 📊 判卷完成！以下是最终成绩单：\n")

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')

    print(f"========== 核心指标 ==========")
    print(f"⭐ 整体准确率 (Accuracy): {acc:.4f}")
    print(f"⭐ 宏平均 F1 (Macro-F1):  {f1:.4f}")
    print("==============================\n")

    print("========== 详细分类报告 ==========")
    print(classification_report(all_labels, all_preds,
                                target_names=["0: Direct (直接)", "1: Intermediate (避重就轻)", "2: Evasive (打太极)"]))


if __name__ == "__main__":
    main()