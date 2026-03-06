# 该文件紧接着01_Entities_Filter.py，对数据计算信息增益、语义相似度等指标，并将结果作为新列保存。

import re
import time
import numpy as np
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_cosine_distances

# 默认路径
INPUT_PATH = "../data/processed/stage1_filtered_data.csv"
OUTPUT_PATH = "../data/processed/stage2_entropy_calculated_data.csv"

# 评分参数
ALPHA_NEW_WORD = 0.65
BETA_LENGTH_NORM = 0.35
MIN_REPLY_LEN = 3
MAX_REPLY_LEN = 500

# 全局预编译正则，避免在 for 循环中反复解析正则树
# 匹配纯标点符号、特殊字符，或者纯数字(包含小数)
INVALID_TOKEN_PATTERN = re.compile(r"^[\W_]+$|^\d+(\.\d+)?$")


def tokenize_zh(text: str) -> list:
    """中文分词并过滤无意义 token"""
    if not isinstance(text, str) or not text:
        return []
    # 使用列表推导式替代 for 循环的 append，执行速度更快
    return [
        t for t in jieba.lcut(text)
        if t.strip() and not INVALID_TOKEN_PATTERN.fullmatch(t)
    ]


def compute_semantic_similarity(df: pd.DataFrame, q_col: str, r_col: str) -> np.ndarray:
    """
    计算 TF-IDF 余弦相似度
    """
    # 避免 apply(" ".join)，列表推导式处理纯字符串拼接极快
    q_strs = [" ".join(tokens) for tokens in df[q_col]]
    r_strs = [" ".join(tokens) for tokens in df[r_col]]

    corpus = q_strs + r_strs

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.95
    )

    print("正在 Fit TF-IDF 矩阵...")
    tfidf = vectorizer.fit_transform(corpus)

    n = len(df)
    q_mat = tfidf[:n]
    r_mat = tfidf[n:]

    print("正在进行向量化余弦相似度计算 (Vectorized)...")
    distances = paired_cosine_distances(q_mat, r_mat)
    return 1.0 - distances


def compute_information_gain(q_tokens: list, r_tokens: list) -> float:
    """计算单个样本的新词比例"""
    if not r_tokens:
        return 0.0
    r_set = set(r_tokens)
    # 直接使用 set.difference() 接收 iterable，C底层执行，无需先将 q 转为 set
    new_words = r_set.difference(q_tokens)
    return len(new_words) / (len(r_set) + 1e-6)


def run_filter():
    start = time.time()
    q_col, r_col = "Qsubj", "Reply"

    print(f"读取数据: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    total = len(df)
    print(f"数据量: {total}")

    # 异常值防御
    df[q_col] = df[q_col].fillna("").astype(str)
    df[r_col] = df[r_col].fillna("").astype(str)

    print("正在进行全局中文分词 (Jieba)...")
    df['q_tokens'] = df[q_col].apply(tokenize_zh)
    df['r_tokens'] = df[r_col].apply(tokenize_zh)

    # 使用纯 Pandas 向量化计算长度惩罚
    print("计算长度惩罚 (Vectorized)...")
    q_len = df[q_col].str.len()
    r_len = df[r_col].str.len()
    ratio = r_len / (q_len + 1e-6)

    # 初始化默认得分为 1.0
    length_scores = np.ones(total)
    # 按条件覆盖 (类似于 numpy 的 if-else 矩阵版)
    length_scores = np.where(ratio > 3.0, 0.8, length_scores)
    length_scores = np.where(ratio > 6.0, 0.4, length_scores)

    # 超出绝对边界的得分为 0.0
    out_of_bounds = (r_len < MIN_REPLY_LEN) | (r_len > MAX_REPLY_LEN)
    df['length_score'] = np.where(out_of_bounds, 0.0, length_scores)

    # 使用 zip + 列表推导式替代 apply(axis=1) 计算词汇增益
    print("计算词汇增益 (List Comprehension)...")
    df['new_word_ratio'] = [
        compute_information_gain(q, r)
        for q, r in zip(df['q_tokens'], df['r_tokens'])
    ]

    # 计算 Info Gain Score
    df['info_gain_score'] = ALPHA_NEW_WORD * df['new_word_ratio'] + BETA_LENGTH_NORM * df['length_score']

    print("计算语义相似度...")
    df["semantic_similarity"] = compute_semantic_similarity(df, 'q_tokens', 'r_tokens')

    # 计算最终得分，仅作为分析列保存，不参与过滤
    df["final_score"] = df["info_gain_score"] * (1.0 - df["semantic_similarity"])

    output_df = df.drop(columns=['q_tokens', 'r_tokens']).copy()
    output_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    cost = time.time() - start
    print(f"计算完成，耗时 {cost:.2f}s")
    print(f"已保存结果: {OUTPUT_PATH}")
    print(f"输出数据量: {len(output_df)} / {total} ({len(output_df) / max(total, 1):.2%})")


if __name__ == "__main__":
    run_filter()