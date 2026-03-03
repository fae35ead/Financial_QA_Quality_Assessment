import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import confusion_matrix

# ==========================================
# 0. 解决 Matplotlib 中文乱码问题 (Windows 必配)
# ==========================================
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# ==========================================
# 1. 动态获取路径
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

test_path = os.path.join(project_root, 'data', 'processed', 'test_dataset.csv')
final_model_dir = os.path.join(project_root, 'models', 'final_bert_qa_model')
# 图片最终保存在项目根目录下，方便你放到 GitHub Readme 里
output_image_path = os.path.join(project_root, 'confusion_matrix.png')

# ==========================================
# 2. 加载模型与测试集数据
# ==========================================
print("正在加载最佳模型和分词器...")
tokenizer = AutoTokenizer.from_pretrained(final_model_dir)
model = AutoModelForSequenceClassification.from_pretrained(final_model_dir)

print("正在读取 10,000 条测试集数据...")
df_test = pd.read_csv(test_path)
df_test = df_test[['model_input', 'final_label']].rename(columns={'final_label': 'label'})
test_dataset = Dataset.from_pandas(df_test)

# Tokenize 处理
def tokenize_function(examples):
    return tokenizer(
        examples["model_input"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

print("正在对测试集进行 Tokenize 处理...")
tokenized_test = test_dataset.map(tokenize_function, batched=True)
tokenized_test = tokenized_test.remove_columns(["model_input"])
tokenized_test.set_format("torch")

# ==========================================
# 3. 极速批量推理 (利用 Trainer 框架)
# ==========================================
# 借用 Trainer 强大的 batch 推理能力，比写 for 循环快得多
training_args = TrainingArguments(
    output_dir="./tmp",
    per_device_eval_batch_size=32, # 仅推理不占太多显存，可以开大点
    report_to="none"               # 关掉没用的日志汇报
)
trainer = Trainer(model=model, args=training_args)

print("🚀 正在让模型对 10,000 道测试题进行闭卷考试，请稍候...")
predictions = trainer.predict(tokenized_test)

# 获取预测的标签 (概率最大的那个类) 和 真实的标签
y_pred = np.argmax(predictions.predictions, axis=-1)
y_true = predictions.label_ids

# ==========================================
# 4. 绘制并保存极其专业的混淆矩阵热力图
# ==========================================
print("🎨 考试结束！正在绘制混淆矩阵...")
cm = confusion_matrix(y_true, y_pred)

# 定义坐标轴的类标
class_names = ['0: 低质量\n(答非所问/水贴)', '1: 中等质量\n(中规中矩)', '2: 高质量\n(干货满满/逻辑清晰)']

plt.figure(figsize=(9, 7))
# 使用 seaborn 画热力图，cmap='Blues' 是一种非常高级、学术的蓝色调
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            annot_kws={"size": 14}) # 格子里的数字大小

plt.title('BERT 问答质量打分模型 - 混淆矩阵 (测试集)', fontsize=16, pad=20)
plt.ylabel('人类/规则 真实标签 (True Label)', fontsize=14)
plt.xlabel('模型 预测标签 (Predicted Label)', fontsize=14)
plt.xticks(fontsize=11)
plt.yticks(fontsize=11, rotation=0)

# 紧凑布局并保存高清大图
plt.tight_layout()
plt.savefig(output_image_path, dpi=300) # dpi=300 保证图片超高清
plt.close()

print(f"\n混淆矩阵高清大图已保存至: {output_image_path}")