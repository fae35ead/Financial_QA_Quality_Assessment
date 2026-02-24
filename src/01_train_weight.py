import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# 直接复用你现有的函数库
from functions import *


def train_and_get_weights(file_path):
    print("1. 正在读取人工打分数据集...")
    try:
        df = pd.read_excel(file_path)
    except:
        df = pd.read_csv(file_path, encoding='utf-8')

    print(f"成功读取 {len(df)} 条人工打分样本！\n")

    # ==========================================
    # 步骤 A: 基础清洗 (与你的 main.py 保持完全一致)
    # 假设你的打分表中，提问列名是'Qsubj'，回答列名是'Reply'
    # 如果是你自己新建的列名（比如'问题'、'回答'），请在这里替换
    # ==========================================
    q_col = 'Qsubj' if 'Qsubj' in df.columns else '问题'
    a_col = 'Reply' if 'Reply' in df.columns else '回答'

    df['clean_q'] = df[q_col].apply(advanced_clean)
    df['clean_a'] = df[a_col].apply(advanced_clean)

    print("2. 正在提取 6 大核心特征...")
    # 1. 信息熵 (Entropy)
    df['F1_entropy'] = df['clean_a'].apply(calc_entropy)

    # 2. 数字密度 (Num Density)
    df['F2_num_density'] = df['clean_a'].apply(calc_num_density)

    # 3. 内容丰富度 (直接使用 clean_a 的长度)
    df['F3_len_a'] = df['clean_a'].apply(len)

    # 4. 相关性 (Relevance)
    df['F4_relevance'] = df.apply(lambda row: calc_relevance(row['clean_q'], row['clean_a']), axis=1)

    # 5. 情感倾向/确定性 (Sentiment)
    df['F5_sentiment'] = df['clean_a'].apply(calc_sentiment)

    # 6. 时效性 (Response Hours) - 调用 get_time_diff
    # 确保你的数据里有 Qtm 和 Reply 时间列，如果没有，这一步需要跳过或给默认值
    df['F6_response_hours'] = df.apply(get_time_diff, axis=1)

    # ==========================================
    # 步骤 B: 数据预处理与归一化
    # ==========================================
    features = ['F1_entropy', 'F2_num_density', 'F3_len_a', 'F4_relevance', 'F5_sentiment', 'F6_response_hours']

    # 异常值处理：剔除时间解析失败的 -1，用中位数填充
    for col in features:
        df[col] = df[col].replace(-1, np.nan)
        df[col] = df[col].fillna(df[col].median())

    X = df[features]
    Y = df['人工评分']  # 确保你的 Excel 中有这一列

    # 归一化：时效性是越小越好，所以在归一化前取负数进行逻辑翻转
    X_copy = X.copy()
    X_copy['F6_response_hours'] = -X_copy['F6_response_hours']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_copy)

    # ==========================================
    # 步骤 C: 训练多元线性回归模型并输出权重
    # ==========================================
    print("3. 正在训练多元线性回归模型...\n")
    model = LinearRegression()
    model.fit(X_scaled, Y)

    # 提取系数并转化为百分比权重
    coefficients = model.coef_
    abs_coefs = np.abs(coefficients)
    weights_pct = (abs_coefs / np.sum(abs_coefs)) * 100

    print("==================================================")
    print("🎯 机器学习倒推【特征权重】结果出炉：")
    print("==================================================")

    result = pd.DataFrame({
        '特征维度': ['信息熵', '数字密度', '内容丰富度(长度)', '相关性', '情感确定性', '时效性'],
        '回归系数': coefficients,
        '重要性占比 (%)': weights_pct
    }).sort_values(by='重要性占比 (%)', ascending=False)

    for _, row in result.iterrows():
        direction = "正相关 ↗" if row['回归系数'] > 0 else "负相关 ↘"
        print(f"[{row['特征维度']:<10}] \t占比: {row['重要性占比 (%)']:>5.2f}% \t({direction})")


if __name__ == "__main__":
    # 指定你打好分的测试集文件路径
    TEST_DATA_PATH = '随机抽取50个问答数据.xlsx'
    train_and_get_weights(TEST_DATA_PATH)