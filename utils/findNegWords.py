import jieba
import pandas as pd
from collections import Counter
import os
import functions

# 确保文件名与你磁盘上的实际文件名完全一致
file_path = '用户与公司问答-2021.xlsx'

# 检查文件是否存在
if not os.path.exists(file_path):
    print(f"错误：找不到文件 '{file_path}'")
else:
    try:
        # 修改这里：读取 .xlsx 文件应使用 read_excel，而不是 read_csv
        # read_excel 不需要指定 encoding，且通常需要 openpyxl 库作为引擎
        df = pd.read_excel(file_path)

        # 筛选出回复长度小于 20 个字的行
        # 注意：这里加了 strConversion 确保 Reply 列是字符串，防止非字符串数据导致 len() 报错
        short_replies = df[df['Reply'].astype(str).apply(lambda x: len(x) < 20)]['Reply']

        # 对这些短回复进行分词
        all_words = []
        for reply in short_replies:
            words = jieba.cut(str(reply))
            all_words.extend(words)

        # 统计词频
        word_counts = Counter(all_words)

        # 剔除停用词
        stop_words = functions.get_stopwords()
        filtered_word_counts = {word: count for word, count in word_counts.items() if
                                word not in stop_words and len(word.strip()) > 1}

        # 打印频率最高的前 30 个词
        top_30_words = Counter(filtered_word_counts).most_common(30)
        print("频率最高的前 30 个词：")
        for word, count in top_30_words:
            print(f"{word}: {count}")

    except Exception as e:
        print(f"处理文件时发生错误: {e}")
