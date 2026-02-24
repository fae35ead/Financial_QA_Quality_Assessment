import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

# 1. 明确我们要使用的预训练模型基座
# 既然是中文 A 股数据，我们选用最经典的中文 BERT
MODEL_NAME = "bert-base-chinese"

# 2. 初始化分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 3. 将 Pandas DataFrame 转换为 HuggingFace 原生的 Dataset 格式
def load_hf_dataset(file_path):
    df = pd.read_csv(file_path)
    # 我们只需要 model_input (文本) 和 final_label (标签)
    df = df[['model_input', 'final_label']]
    # HuggingFace 默认分类任务的标签列名叫做 'label'
    df = df.rename(columns={'final_label': 'label'})
    return Dataset.from_pandas(df)


# 加载三个数据集
train_dataset = load_hf_dataset('train_dataset.csv')
valid_dataset = load_hf_dataset('valid_dataset.csv')
test_dataset = load_hf_dataset('test_dataset.csv')

print(f"训练集规模: {len(train_dataset)}")


# Token化 映射函数
def tokenize_function(examples):
    return tokenizer(
        examples["model_input"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

# 批量应用 token化 (batched=True 极大地提升处理速度)
print("正在对数据进行 Tokenize 处理...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_valid = valid_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# 深度学习模型不需要看原始文本，只需要看 input_ids 等张量特征和 label
# 我们把不需要的列丢弃掉
columns_to_remove = ["model_input"]
tokenized_train = tokenized_train.remove_columns(columns_to_remove)
tokenized_valid = tokenized_valid.remove_columns(columns_to_remove)
tokenized_test = tokenized_test.remove_columns(columns_to_remove)

# 设置数据格式为 PyTorch 的 tensor
tokenized_train.set_format("torch")
tokenized_valid.set_format("torch")
tokenized_test.set_format("torch")

print("数据管道打通！准备喂给模型。")