import pandas as pd
import glob

# 1. 一键读取你跑出来的 4 个 CSV 文件并合并
# 以 'advanced_features_' 开头
files = glob.glob('advanced_features_*.csv')
df_all = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

print(f"成功合并 {len(files)} 个文件，总数据量：{len(df_all)} 条")

# 2. 扔掉之前各自为战的“局部标签”和“局部归一化分数”
cols_to_drop = [c for c in df_all.columns if '_norm' in c or 'score' in c or 'label' in c]
df_all = df_all.drop(columns=cols_to_drop)

# 3. 在全局数据上，统一进行归一化
for col in ['entropy', 'num_density', 'len_a', 'relevance']:
    df_all[f'{col}_norm'] = (df_all[col] - df_all[col].min()) / (df_all[col].max() - df_all[col].min())

df_all['time_score'] = df_all['response_hours'].apply(lambda x: 100 if x < 0 else x)
df_all['time_norm'] = 1 - (df_all['time_score'] - df_all['time_score'].min()) / (df_all['time_score'].max() - df_all['time_score'].min())

# 4. 使用之前机器学习算出来的完美权重，计算全局得分
df_all['quality_score'] = (0.42 * df_all['entropy_norm'] +
                           0.23 * df_all['relevance_norm'] +
                           0.15 * df_all['num_density_norm'] +
                           0.13 * df_all['time_norm'] -
                           0.06 * df_all['len_a_norm'])

# 5. 在全局 5 万条数据中，划出真正的 Top 30% 和 Bottom 30%
high_threshold = df_all['quality_score'].quantile(0.7)
low_threshold = df_all['quality_score'].quantile(0.3)

def label_quality(score):
    if score >= high_threshold: return 2
    if score <= low_threshold: return 0
    return 1

df_all['final_label'] = df_all['quality_score'].apply(label_quality)

# 6. 保存为最终的全局大一统特征集
df_all.to_csv('advanced_features_global_all.csv', index=False)
print("全局打标完成！这才是工业级的一致性数据集！")