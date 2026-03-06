# 该文件的作用是从你之前计算的全量特征数据中，按照 final_score 的分位数进行分层抽样，最终生成一个包含 1000 条高质量金标准待打标数据的 CSV 文件，供大模型进行打标使用。

import pandas as pd

df = pd.read_csv("../data/processed/stage2_entropy_calculated_data.csv") # 你保存全量特征的文件

# 策略：利用你之前算出的 final_score 分位数，将数据强行切分成 5 个难度桶，每个桶抽 200 条
# 这样不仅覆盖了“一眼假”的废话，也覆盖了“包装极好的太极拳”和“简短的高质回答”
bins = pd.qcut(df['final_score'], q=5, labels=['Bin1', 'Bin2', 'Bin3', 'Bin4', 'Bin5'])
df['score_bucket'] = bins

sampled_df = df.groupby('score_bucket').apply(lambda x: x.sample(200, random_state=42)).reset_index(drop=True)

# 仅保留需要给大模型看的内容，打乱顺序防止大模型产生位置偏见
golden_df = sampled_df[['Qsubj', 'Reply', 'final_score']].sample(frac=1, random_state=1024).reset_index(drop=True)
golden_df.to_csv("../data/processed/golden_samples_to_label.csv", index=False, encoding='utf-8-sig')
print("✅ 分层抽样完成，共提取 1000 条高质量金标准待打标数据！")