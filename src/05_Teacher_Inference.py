'''该文件的主要功能是使用之前训练好的 Teacher 模型对全量无标签数据进行推理，生成软标签（概率分布），并将这些软标签与原始数据合并保存为新的 CSV 文件，为后续的学生模型训练提供丰富的监督信号。'''

import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# ================= 配置区 =================
# 这里填你之前清洗好的、包含70万条问答的全量数据文件路径
UNLABELED_DATA_FILE = "../data/processed/stage2_entropy_calculated_data.csv"
OUTPUT_FILE = "../data/processed/soft_labeled_100k.csv"
TEACHER_MODEL_DIR = "../models/teacher_cross_encoder"  # 刚才训练好的模型路径

SAMPLE_SIZE = 100000  # 100k数据进行蒸馏，提升速度、降低资源占用
MAX_LEN = 384  # 和训练时保持一致
BATCH_SIZE = 32  # 纯推理不计算梯度，RTX 2070S (8GB) 开到 32 甚至 64 毫无压力


class QADataset(Dataset):
    """自定义 PyTorch 数据集，用于高效批量加载数据"""

    def __init__(self, texts_q, texts_r, tokenizer, max_len):
        self.texts_q = texts_q
        self.texts_r = texts_r
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts_q)

    def __getitem__(self, item):
        q = str(self.texts_q[item])
        r = str(self.texts_r[item])

        # 依然使用 Cross-Encoder 的拼接方式：[CLS] Q [SEP] R [SEP]
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
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).flatten()
        }


def main():
    print("1. 正在加载全量无标签数据...")
    # 假设你的列名叫 'Question' 和 'Reply'，如果叫 'Qsubj' 请自行修改
    df_all = pd.read_csv(UNLABELED_DATA_FILE)

    # 【核心逻辑】：随机抽取 2 万条作为蒸馏数据集
    print(f"原始数据总量: {len(df_all)} 条。开始随机抽取 {SAMPLE_SIZE} 条...")
    df_sample = df_all.sample(n=SAMPLE_SIZE, random_state=42).copy()
    df_sample.reset_index(drop=True, inplace=True)

    # 提取文本列表
    questions = df_sample['Qsubj'].tolist()
    replies = df_sample['Reply'].tolist()

    print(f"\n2. 正在加载训练好的 Teacher 模型和分词器...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前使用计算设备: {device}")

    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(TEACHER_MODEL_DIR)
    model.to(device)
    model.eval()  # 【极其重要】切换到评估模式，关闭 Dropout

    print("\n3. 构建高速数据加载器 (DataLoader)...")
    dataset = QADataset(questions, replies, tokenizer, MAX_LEN)
    # num_workers 设置为 0 避免 Windows 下的多进程报错，如果你用 Linux 可以调高
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 用于存放所有预测概率的列表
    all_probs = []

    print("\n4. 🚀 开启极速推理模式 (生成 Soft Labels)...")
    # torch.no_grad() 告诉 PyTorch 不要记录梯度，显存占用骤降，速度飙升！
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Teacher 打标进度"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            # 模型前向传播，得到原始的 Logits (未归一化的分数)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            logits = outputs.logits

            # 【知识蒸馏核心】：使用 Softmax 将 Logits 转换为 0~1 之间的概率分布
            probs = torch.nn.functional.softmax(logits, dim=-1)

            # 将 GPU 上的张量拉回 CPU 并转为 numpy 数组
            all_probs.extend(probs.cpu().numpy())

    print("\n5. 正在将软标签合并回数据并保存...")
    all_probs = np.array(all_probs)

    # 新增三列概率分布特征
    df_sample['prob_0'] = all_probs[:, 0]
    df_sample['prob_1'] = all_probs[:, 1]
    df_sample['prob_2'] = all_probs[:, 2]

    # 顺便给一个硬标签（概率最大的一项），方便后续你自己人工抽检看效果
    df_sample['teacher_hard_label'] = np.argmax(all_probs, axis=1)

    df_sample.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\n2万条数据的软标签提取完毕！")
    print(f"文件已保存至: {OUTPUT_FILE}")

    # 打印几条看看效果
    print("\n=== 软标签效果预览 (前3条) ===")
    print(df_sample[['teacher_hard_label', 'prob_0', 'prob_1', 'prob_2']].head(3))


if __name__ == "__main__":
    main()