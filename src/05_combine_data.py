import pandas as pd
import glob
import os

# 1. 使用动态绝对路径寻找文件
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

search_folder = os.path.join(project_root, 'data', 'processed')
search_pattern = os.path.join(search_folder, 'advanced_features_*.csv')

files = glob.glob(search_pattern)

# 加一个安全拦截，防止再次出现空列表报错
if len(files) == 0:
    print(f"在路径 {search_pattern} 下没有找到任何文件！请检查文件到底存在哪了。")
    exit()

print(f"✅ 成功找到 {len(files)} 个文件，准备合并...")
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
output_path = os.path.join(project_root, 'data', 'processed', 'advanced_features_global_all.csv')
df_all.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"全局特征表已保存到 {output_path}，总数据量：{len(df_all)} 条")