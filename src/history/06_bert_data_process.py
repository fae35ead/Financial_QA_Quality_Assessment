import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 1. 动态获取路径并加载刚才生成的全局数据
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
global_file_path = os.path.join(project_root, 'data', 'processed', 'advanced_features_global_all.csv')

print("正在读取全局数据集...")
df = pd.read_csv(global_file_path)

# 2. 构造 BERT 的输入文本 (非常关键的一步)
# 我们使用 BERT 的特殊分隔符 [SEP] 把问题和回答无缝拼接起来
if 'model_input' not in df.columns:
    df['model_input'] = df['clean_q'].astype(str) + "[SEP]" + df['clean_a'].astype(str)

# 3. 清理多余列，只保留模型需要的输入和标签
df = df[['model_input', 'final_label']]

# 4. 科学划分数据集 (80% 训练集, 10% 验证集, 10% 测试集)
# stratify=df['final_label'] 保证了切分后各类别的比例和原表一致，防止模型学偏
print("正在划分训练集、验证集和测试集...")
train_df, temp_test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['final_label'])
valid_df, test_df = train_test_split(temp_test_df, test_size=0.5, random_state=42, stratify=temp_test_df['final_label'])

# 5. 保存切分好的数据集到 data/processed 目录
train_path = os.path.join(project_root, 'data', 'processed', 'train_dataset.csv')
valid_path = os.path.join(project_root, 'data', 'processed', 'valid_dataset.csv')
test_path = os.path.join(project_root, 'data', 'processed', 'test_dataset.csv')

train_df.to_csv(train_path, index=False)
valid_df.to_csv(valid_path, index=False)
test_df.to_csv(test_path, index=False)

print("\n✅ 数据集准备完毕，可以喂给 BERT 了！")
print(f"训练集: {len(train_df)} 条 -> 已保存至 {train_path}")
print(f"验证集: {len(valid_df)} 条 -> 已保存至 {valid_path}")
print(f"测试集: {len(test_df)} 条 -> 已保存至 {test_path}")