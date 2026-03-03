import pandas as pd
import numpy as np
import re
import jieba
import os
import math
from datetime import datetime

from utils.functions import advanced_clean, calc_entropy, calc_num_density, calc_relevance, get_time_diff, label_quality_generic


# === 4. 主处理流程 ===
if __name__ == "__main__":
    # 1. 动态定位：获取当前文件的目录，再推导到项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # 2. 拼接出目标数据的绝对路径
    file_path = os.path.join(project_root, 'data', 'raw', '用户与公司问答-2010.xlsx')
    try:
        df = pd.read_excel(file_path)
    except:
        # 如果读取失败，尝试读取 csv 或者报错
        print("Excel读取失败，请检查文件")
        exit()

    # 基础清洗
    df['clean_q'] = df['Qsubj'].apply(advanced_clean)
    df['clean_a'] = df['Reply'].apply(advanced_clean)
    df = df[(df['clean_q'].str.len() > 5) & (df['clean_a'].str.len() > 2)].copy()

    print("正在进行特征工程...")

    # --- 维度 1: 内容特征 ---
    # 计算回答的信息熵
    df['entropy'] = df['clean_a'].apply(calc_entropy)
    # 计算回答的数字密度
    df['num_density'] = df['clean_a'].apply(calc_num_density)
    # 计算回答长度
    df['len_a'] = df['clean_a'].apply(len)
    # 计算相关性
    df['relevance'] = df.apply(lambda row: calc_relevance(row['clean_q'], row['clean_a']), axis=1)
    # 计算情感倾向/确定性
    # df['sentiment'] = df['clean_a'].apply(calc_sentiment)
    # 只占比1.64%，故不纳入综合打分

    # --- 维度 2: 时效性特征 ---
    df['response_hours'] = df.apply(get_time_diff, axis=1)

    # --- 维度 3: 综合打分 (Heuristic Scoring) ---
    # 先进行归一化 (Min-Max Normalization) 以便加权
    for col in ['entropy', 'num_density', 'len_a', 'relevance']:
        min_val = df[col].min()
        max_val = df[col].max()
        # 加上 1e-9 防止除以 0
        df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val + 1e-9)

    # 时效性是越短越好，所以要反向归一化
    df['time_score'] = df['response_hours'].apply(lambda x: 100 if x < 0 else x)  # 处理异常
    df['time_norm'] = 1 - (df['time_score'] - df['time_score'].min()) / (df['time_score'].max() - df['time_score'].min())

    # === 核心：定义高质量标准 ===
    # 信息熵(42%) + 相关性(23%) + 数字密度(15%) + 时效性(13%) - 长度惩罚(6%)
    df['quality_score'] = (0.42 * df['entropy_norm'] +
                           0.23 * df['relevance_norm'] +
                           0.15 * df['num_density_norm'] +
                           0.13 * df['time_norm'] -
                           0.06 * df['len_a_norm'])  # 注意这里是减号，惩罚又长又废话的公关稿

    # 设定阈值：前 30% 为高质量，后 30% 为低质量
    high_threshold = df['quality_score'].quantile(0.7)
    low_threshold = df['quality_score'].quantile(0.3)


    def label_quality(score):
        # 使用 functions.py 中的通用函数，或者直接在这里重写
        return label_quality_generic(score, high_threshold, low_threshold)


    df['final_label'] = df['quality_score'].apply(label_quality)

    # === 5. 结果展示 ===
    print("\n=== 特征计算结果 ===")
    print(df[['clean_a', 'entropy', 'num_density', 'response_hours', 'quality_score', 'final_label']].head())

    print("\n=== 高质量回答示例 ===")
    print(df[df['final_label'] == 2]['clean_a'].head(3).values)

    print("\n=== 低质量回答示例 ===")
    print(df[df['final_label'] == 0]['clean_a'].head(3).values)

    # 保存带有特征和标签的数据，供后续分析
    # 动态拼接输出路径，保持项目结构整洁
    output_path = os.path.join(project_root, 'data', 'processed', 'advanced_features_2010.csv')
    df.to_csv(output_path, index=False)
    print(f"\n特征提取完毕！已保存至: {output_path}")
