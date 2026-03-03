import pandas as pd
import numpy as np
import re
import jieba
import math
from datetime import datetime
import os

# === 1. 加载停用词 ===
import os


def get_stopwords(file_name='scu_stopwords.txt'):
    base_stopwords = set()

    # 1. 动态定位：获取当前文件 (functions.py) 的目录，再推导到项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # 2. 拼接出目标停用词表的绝对路径
    file_path = os.path.join(project_root, 'data', 'others', '停用词词表', file_name)

    try:
        # 直接判断这一个绝对路径是否存在即可，不需要再写 elif 猜路径了
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    word = line.strip()
                    if word:
                        base_stopwords.add(word)
            print(f"成功加载本地停用词表，共 {len(base_stopwords)} 个词。")
        else:
            print(f"未找到停用词文件，路径为 {file_path}")

    except Exception as e:
        print(f"加载停用词失败: {e}")

    # 领域特定客套话 (金融/QA问答领域的特有停用词，这部分你写得非常好！)
    domain_stopwords = {'感谢', '关注', '您好', '谢谢', '提问', '投资者', '公司', '公告', '敬请', '注意', '风险',
                        '查阅', '请教', '请问', '尊敬', '董秘', '谢谢您', '感谢您', '你好', '关心', '支持', '不便'}

    # 将本地读取的停用词和代码里硬编码的停用词合并取并集
    final_stopwords = base_stopwords.union(domain_stopwords)
    print(f"合并特定客套话后，最终停用词库共 {len(final_stopwords)} 个词。\n")

    return final_stopwords

STOPWORDS = get_stopwords()

# === 2. 高级清洗函数 ===
def advanced_clean(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\s+', '', text)
    return text

# === 3. 学术级特征计算函数 ===

# 信息熵计算：衡量回答内容的多样性和信息量
def calc_entropy(text):
    words = [w for w in jieba.cut(text) if w not in STOPWORDS and len(w) > 1]
    if not words: return 0
    data_set = list(set(words))
    freq_list = []
    for entry in data_set:
        counter = 0.0
        for word in words:
            if word == entry:
                counter += 1
        freq_list.append(counter)

    entropy = 0.0
    for freq in freq_list:
        p = freq / len(words)
        entropy -= p * math.log(p, 2)
    return entropy

# 数字密度计算：衡量回答中数字信息的丰富程度
def calc_num_density(text):
    nums = re.findall(r'\d+(?:\.\d+)?%?', text)
    return len(nums) / (len(text) + 1)

# 特异词密度计算：衡量废话多少
def calc_unique_ratio(text):
    words = [w for w in jieba.cut(text) if w not in STOPWORDS]
    if not words: return 0
    return len(set(words)) / len(words)

# 相关性计算：衡量回答与问题的相关程度
def calc_relevance(q_text, a_text):
    q_set = set([w for w in jieba.cut(q_text) if w not in STOPWORDS])
    a_set = set([w for w in jieba.cut(a_text) if w not in STOPWORDS])
    if not q_set or not a_set:
        return 0
    intersection = q_set.intersection(a_set)
    return len(intersection) / (len(q_set) + len(a_set) - len(intersection))

# 时效性计算：衡量回答的及时程度（小时为单位）
def get_time_diff(row):
    try:
        t1 = pd.to_datetime(row['Qtm'])
        t2 = pd.to_datetime(row['Recvtm'])
        if pd.isna(t1) or pd.isna(t2): return -1
        return (t2 - t1).total_seconds() / 3600.0
    except:
        return -1

# 情感倾向计算：衡量回答中确定和不确定的词汇比例，反映回答的信心程度
def calc_sentiment(text):
    positive_words = {'已完成', '明确', '预计', '签署', '完成', '达成', '确定', '成功', '实现', '增长', '提升', '改善', '优化', '推进', '加速', '稳步', '持续', '良好', '盈利'}
    negative_words = {'不确定', '可能', '以公告为准', '无法预测', '受多重因素影响', '推迟', '延期', '不排除', '存在风险', '挑战', '困难', '目前', '尚未', '正在', '尚无', '不便'}

    pos_count = sum(text.count(word) for word in positive_words)
    neg_count = sum(text.count(word) for word in negative_words)

    if pos_count + neg_count == 0:
        return 0
    return (pos_count - neg_count) / (pos_count + neg_count)

# 质量标签函数：根据综合得分设定高质量、中等质量、低质量标签
def label_quality_generic(score, high, low):
    if score >= high: return 2
    if score <= low: return 0
    return 1


