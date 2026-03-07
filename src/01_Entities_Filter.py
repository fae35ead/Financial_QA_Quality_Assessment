'''该文件实现了第一层过滤：实体存在性过滤，核心思路是通过正则剥离董秘回答中的套话部分，提取核心“肉饼”，
然后使用 FlashText 实体识别引擎扫描核心文本中是否存在金融实体词汇。如果剥离后的核心文本长度过短（例如不到 5 个字），或者没有提取到任何金融实体，则判定该回答为废话，予以剔除。通过这一层过滤，
我们可以大幅度降低后续模型训练和推理的噪音水平，提升整体数据质量和模型性能。
'''

import pandas as pd
import re
import time
from flashtext import KeywordProcessor

# 1. 规则与正则定义

# 头部套话正则
prefix_pattern = re.compile(
    r'^(?:'
    r'(?:尊敬的)?(?:投资者|股东)[，。！\s]*|'
    r'(?:尊敬的)?(?:投资者|股东)?您好[，。！\s]*|'
    r'(?:非常)?(?:感谢|谢谢|多谢)您?(?:对公司)?的?(?:关注|关心|支持|建议|提问)(?:[与和]?(?:关注|关心|支持|建议|提问))?[，。！\s]*'
    r')+'
)

# 尾部套话正则
suffix_pattern = re.compile(
    r'(?:'
    r'(?:非常)?(?:感谢|谢谢|多谢)您?(?:对公司)?的?(?:关注|关心|支持|建议|提问)(?:[与和]?(?:关注|关心|支持|建议|提问))?[，。！\s]*|'
    r'请您关注公司后续的公告[，。！\s]*|'
    r'敬请谅解[，。！\s]*(?:谢谢[，。！\s]*)?|'
    r'具体(?:内容|情况|数据|信息)?请以.*?公告为准[，。！\s]*|'
    r'谢谢[，。！\s]*'
    r')+$'
)


# 2. 核心处理函数

def trim_boilerplate(text):
    """清洗董秘回答中的头尾套话"""
    if not isinstance(text, str):
        return ""
    text = re.sub(prefix_pattern, '', text)
    text = re.sub(suffix_pattern, '', text)
    return text.strip()


def build_financial_processor():
    """初始化并装载 FlashText 实体识别引擎"""
    keyword_processor = KeywordProcessor()
    dict_files = ['../data/others/THUOCL_caijing.txt',
                  '../data/others/baostock_entities.txt',
                  '../data/others/custom_entities.txt']

    total_words = 0
    for file_path in dict_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                words = [line.strip() for line in f if len(line.strip()) > 1]
                keyword_processor.add_keywords_from_list(words)
                total_words += len(words)
        except FileNotFoundError:
            print(f"警告: 找不到实体词典文件 {file_path}")

    print(f"FlashText 引擎初始化完成，共装载 {total_words} 个金融实体！")
    return keyword_processor


def check_entity_existence(row_text, processor, min_length=5):
    """
    实体存在性单行判定逻辑：
    返回 True (保留) 或 False (剔除)
    """
    # 1. 剥离套话提取核心“肉饼”
    core_text = trim_boilerplate(row_text)

    # 2. 长度兜底：如果剥离后连 5 个字都不到了，绝对是废话
    if len(core_text) < min_length:
        return False

    # 3. 实体扫描：如果提取到的实体列表为空，也是废话
    return len(processor.extract_keywords(core_text)) > 0


# ==================== 3. 主 Pipeline 流程 ====================

def main():
    processor = build_financial_processor()

    input_path = '../data/processed/cleaned_data.csv'
    output_path = '../data/processed/stage1_filtered_data.csv'
    rejected_output_path = '../data/processed/stage1_rejected_data.csv'

    SAVE_REJECTED = True  # 可选开关，用于保存过滤的数据

    print(f"正在读取数据: {input_path}")
    df = pd.read_csv(input_path)
    initial_count = len(df)
    print(f"原始数据量: {initial_count} 行")

    print("开始执行【第一层：实体存在性过滤】...")
    start_time = time.time()

    # 1. 轻量预处理：向量化填充空值并转为字符串，极大降低 apply 内部异常
    reply_series = df['Reply'].fillna('').astype(str)

    # 2. 执行核心过滤逻辑
    is_valid_mask = reply_series.apply(lambda x: check_entity_existence(x, processor))

    # 3. 数据分流
    df_filtered = df[is_valid_mask].copy()

    end_time = time.time()
    passed_count = len(df_filtered)
    drop_count = initial_count - passed_count

    print(f"过滤完成！耗时: {end_time - start_time:.2f} 秒")
    print(f"拦截废话数据: {drop_count} 行")
    print(f"剩余有效数据: {passed_count} 行 (通过率 {passed_count / initial_count * 100:.2f}%)")

    # 4. 结果保存
    df_filtered.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"第一阶段过滤数据已保存至: {output_path}")

    if SAVE_REJECTED:
        df_rejected = df[~is_valid_mask].copy()
        df_rejected.to_csv(rejected_output_path, index=False, encoding='utf-8-sig')
        print(f"被拦截数据已保存至: {rejected_output_path}")

if __name__ == '__main__':
    main()