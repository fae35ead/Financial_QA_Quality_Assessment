'''该文件负责从原始数据文件夹中读取所有 Excel 文件，合并成一个 DataFrame，并进行清洗处理。清洗步骤包括：
1. 剔除空回答：删除 'Reply' 或 'Qsubj' 列中有空值的行。
2. 剔除时间错乱数据：如果 'Qtm' 和 'Recvtm' 列存在，删除 'Recvtm' 早于 'Qtm' 的行。
3. 向量化文本清洗：使用 pandas 的字符串方法批量去除 'Reply' 和 'Qsubj' 列中的 HTML 标签和多余空白字符。
清洗完成后，数据将被保存为 CSV 格式，以突破 Excel 104万行的限制。'''

import os
import pandas as pd

# 定义文件夹路径
raw_data_folder = '../data/raw'

# 合并所有 Excel 文件
def load_and_merge_data(folder_path):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.xlsx')]

    # 提示：读取 Excel 依然会比较慢，请耐心等待
    print(f"找到 {len(all_files)} 个 Excel 文件，正在读取...")
    dataframes = [pd.read_excel(file) for file in all_files]
    merged_data = pd.concat(dataframes, ignore_index=True)
    return merged_data

# 清洗函数
def clean_data(df):
    # 1. 剔除空回答
    df = df.dropna(subset=['Reply', 'Qsubj'])

    # 2. 剔除时间错乱数据
    if 'Qtm' in df.columns and 'Recvtm' in df.columns:
        df = df[df['Recvtm'] >= df['Qtm']]

    # 3. 向量化文本清洗 (同时处理 Reply 和 Question)
    for col in ['Reply', 'Qsubj']:
        if col in df.columns:
            df[col] = df[col].astype(str)
            # 使用 pandas 内置的向量化正则替换去 HTML 标签
            df[col] = df[col].str.replace(r'<[^>]+>', '', regex=True)
            # 去除多余空白字符
            df[col] = df[col].str.replace(r'\s+', '', regex=True)

    return df


def main():
    raw_data = load_and_merge_data(raw_data_folder)
    print(f"原始数据加载完成，共 {len(raw_data)} 行。开始清洗...")

    cleaned_data = clean_data(raw_data)
    print(f"清洗完成，剩余 {len(cleaned_data)} 行。正在保存...")

    # 关键修改：保存为 CSV 格式，突破 Excel 104万行的限制
    save_path = '../data/processed/cleaned_data.csv'

    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存为 CSV (如果需要极速读写，可以改为 .to_parquet('...parquet'))
    cleaned_data.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"数据已成功保存到 {save_path}")


if __name__ == '__main__':
    main()