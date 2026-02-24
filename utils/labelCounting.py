import pandas as pd

# 读取生成的 CSV 文件
# 确保文件名与你 main.py 中保存的文件名一致
df_result = pd.read_csv('advanced_features_2021.csv')

print("\n=== 标签分布统计 ===")
# value_counts() 默认按数量降序排列
# sort_index() 可以让它按标签 0, 1, 2 的顺序排列
counts = df_result['final_label'].value_counts().sort_index()

print(counts)

print("\n=== 详细统计 ===")
for label, count in counts.items():
    label_name = {0: "低质量", 1: "中等", 2: "高质量"}.get(label, "未知")
    print(f"标签 {label} ({label_name}): {count} 条")
