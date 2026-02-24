import pandas as pd


def spot_check_labels(file_path='advanced_features_2023.csv', sample_size=5):
    print("正在加载数据进行抽样质检...\n")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"找不到文件 {file_path}，请确认您的特征表文件名是否正确！")
        return

    # 抽取高质量数据 (Label 2)
    # random_state=None 保证你每次运行这段代码，看到的都是不同的数据
    high_quality = df[df['final_label'] == 2].sample(n=sample_size, random_state=None)

    print("==================================================")
    print("【高质量标签 (Label 2)】抽样盲测")
    print("==================================================")
    for i, (_, row) in enumerate(high_quality.iterrows(), 1):
        print(f"\n[优质样本 {i}]")
        print(f"🙋 提问：{row['clean_q']}")
        print(f"💬 回答：{row['clean_a']}")
        print("-" * 50)

    # 抽取低质量数据 (Label 0)
    low_quality = df[df['final_label'] == 0].sample(n=sample_size, random_state=None)

    print("\n\n==================================================")
    print("【低质量标签 (Label 0)】抽样盲测")
    print("==================================================")
    for i, (_, row) in enumerate(low_quality.iterrows(), 1):
        print(f"\n[劣质样本 {i}]")
        print(f"🙋 提问：{row['clean_q']}")
        print(f"💬 回答：{row['clean_a']}")
        print("-" * 50)

if __name__ == '__main__':
    spot_check_labels()