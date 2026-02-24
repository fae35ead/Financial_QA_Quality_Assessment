import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score


def build_baseline_and_split(file_path='advanced_features_global_all.csv'):
    print("1. 正在加载全量特征数据...")
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"找不到文件 {file_path}，请检查路径。")
        return

    # 处理可能的空值
    df['clean_q'] = df['clean_q'].fillna('')
    df['clean_a'] = df['clean_a'].fillna('')

    # 【核心操作】构建模型输入：将“提问”和“回答”拼接，让模型能同时看到上下文
    df['model_input'] = "问：" + df['clean_q'].astype(str) + " 答：" + df['clean_a'].astype(str)

    # 提取需要的列：输入文本 和 质量标签
    data = df[['model_input', 'final_label']].copy()

    print("2. 正在进行标准数据集切分 (8:1:1)...")
    # 第一次切分：80% 训练集，20% 临时集（包含验证和测试）
    # stratify 保证了切分后各个集合中的 0, 1, 2 标签比例一致
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42, stratify=data['final_label'])

    # 第二次切分：把 20% 的临时集对半劈开，变成 10% 验证集，10% 测试集
    valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42,
                                             stratify=temp_data['final_label'])

    # 保存切分好的标准数据集（这是未来喂给 GPU 和 BERT 的最终口粮！）
    train_data.to_csv('train_dataset.csv', index=False)
    valid_data.to_csv('valid_dataset.csv', index=False)
    test_data.to_csv('test_dataset.csv', index=False)
    print(
        f"切分完成！已保存为:\n - train_dataset.csv ({len(train_data)}条)\n - valid_dataset.csv ({len(valid_data)}条)\n - test_dataset.csv ({len(test_data)}条)")

    print("\n3. 正在训练传统机器学习基线模型 (Baseline: TF-IDF + Logistic Regression)...")
    # 使用 TF-IDF 将文本向量化
    vectorizer = TfidfVectorizer(max_features=20000)

    # 拟合训练集并转换
    X_train = vectorizer.fit_transform(train_data['model_input'])
    # 仅转换测试集（严禁在测试集上 fit，防止数据泄露）
    X_test = vectorizer.transform(test_data['model_input'])

    y_train = train_data['final_label']
    y_test = test_data['final_label']

    # 训练逻辑回归分类器 (速度极快，是工业界首选的 Baseline)
    clf = LogisticRegression(solver='saga', max_iter=2000, n_jobs=-1)
    clf.fit(X_train, y_train)

    # 在从未见过的【测试集】上进行预测
    y_pred = clf.predict(X_test)

    print("\n==================================================")
    print("传统机器学习 TF-IDF + 逻辑回归 Baseline 测试集评估报告")
    print("==================================================")
    print(f"总体准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
    print("\n详细分类报告 (Precision / Recall / F1-Score):")
    print(classification_report(y_test, y_pred, target_names=['0 (低质量)', '1 (中等质量)', '2 (高质量)']))

    print("传统的 TF-IDF 只能识别零散的关键词汇，无法理解句子的深度语义和逻辑。")
    print("这正是我们接下来要引入深度学习 BERT 大模型的根本原因！")


if __name__ == '__main__':
    build_baseline_and_split()