import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 当前炼丹炉已连接至: {device.type.upper()} !!!")
if device.type != 'cuda':
    print("⚠️ 警告：没有检测到 GPU！如果坚持训练可能需要几天几夜。请检查右上角资源类型。")

# 1. 加载带有一个分类头（Classification Head）的预训练 BERT 模型
# num_labels=3 代表我们有 0, 1, 2 三个类别
print("正在下载 BERT 模型权重 (大概需要几百MB)...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

# 2. 定义评估函数：在每个 Epoch 结束后，考考模型学得怎么样
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # logits 是模型输出的概率分布，我们取最大概率的索引作为预测类别
    predictions = np.argmax(logits, axis=-1)

    # 计算精确率、召回率、F1和准确率 (macro代表对三个类别求平均)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro', zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# 3. 核心调参区：定义训练参数 (TrainingArguments)
training_args = TrainingArguments(
    output_dir='./bert_qa_quality_model',
    eval_strategy="epoch",                # 顺应最新版本的参数名
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir='./logs',
    logging_steps=100,
    save_total_limit=2,
    fp16=True,  # 开启半精度混合训练，显存减半，速度翻倍
)


# 4. 组装终极训练器 (Trainer)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,        # 喂入 8 万条训练数据
    eval_dataset=tokenized_valid,         # 喂入验证集进行期中考试
    compute_metrics=compute_metrics,      # 告诉模型怎么算分
)

# 5. 🚀 正式点火！开始训练
print("🚀 点火！开始训练 BERT 大模型...")
trainer.train()

# 6. 🏆 训练结束后，在从未见过的测试集上进行最终考试
print("🏆 训练完成！正在测试集上进行最终评估...")
test_results = trainer.evaluate(tokenized_test)
print(f"最终测试集准确率 (Accuracy): {test_results['eval_accuracy']:.4f}")

# 7. 导出最终模型
final_model_path = "./final_bert_qa_quality_model"
print(f"正在将最佳模型导出至：{final_model_path}")
# 保存模型权重和配置文件
trainer.save_model(final_model_path)
# 必须同步保存 tokenizer，否则以后加载模型时会由于词表不匹配报错
tokenizer.save_pretrained(final_model_path)
print("✅ 模型持久化保存完成。你可以直接下载这个文件夹用于后续推理。")