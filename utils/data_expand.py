# 该文件针对用于训练学生模型时，样本量较少的少数类（如“外部归因”、“战略性模糊”、“财务表现指引”）进行定向数据增强。
# 核心思路是利用 LLM 生成大量的高质量变体，通过替换行业背景、数字细节和事件名称等方式，保持原有的核心逻辑和话术策略不变，同时确保生成的样本依然符合原有的类别特征。
import pandas as pd
import json
import asyncio
import os
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# ================= 配置区 =================
INPUT_FILE = "../data/processed/labeled_subnodes_samples.csv"
OUTPUT_FILE = "../data/processed/synthetic_augmented_samples.csv"

API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "")
CONCURRENCY_LIMIT = 5

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# 我们只对样本量小于 60 的少数类进行精准定向扩充
TARGET_CLASSES = ["外部归因", "战略性模糊", "财务表现指引"]

# 针对不同类别的扩充倍数设定（缺得越多的，生成越多）
AUGMENT_MULTIPLIERS = {
    "外部归因": 20,  # 原来只有4条，每条生成20个变体，总共新增约 80 条
    "战略性模糊": 3,  # 原来37条，每条生成3个变体，新增约 111 条
    "财务表现指引": 2  # 原来40条，每条生成2个变体，新增约 80 条
}

PROMPT_TEMPLATE = """"你是一位专业的 A 股董秘话术模拟专家。
我现在提供给你一个真实的【投资者提问】和【董秘回答】，它属于【{sub_label}】类别。
你的任务是：保持这种核心逻辑和话术策略不变，生成 {n_variants} 个全新的、多样化的变体。

要求：
1. 替换行业背景（例如从新能源换成医药、半导体、消费、化工等）。
2. 替换具体的数字、项目名称或宏观事件（如将“疫情”替换为“地缘政治”、“原材料暴涨”、“海运费波动”等）。
3. 语气要完美符合中国 A 股董秘那种官方、严谨、或者圆滑的口吻。
4. 绝不能改变它的本质类别（必须依然是明显的【{sub_label}】）。

原始提问：{question}
原始回答：{reply}

=== 输出格式 ===
请仅输出一个合法的 JSON 数组，绝不能包含 Markdown 符号（如 ```json）。
【极其重要：防止解析报错的 2 条铁律】
1. 文本内容（Value）中如果需要使用引号，请**务必使用中文的单引号（‘’）或中文双引号（“”）**，绝对不能使用英文双引号（"），否则会彻底破坏 JSON 结构！
2. 数组中的每个对象之间，必须用标准的英文逗号（,）隔开！

示例格式：
[
  {{"Question": "新的提问1", "Reply": "新的回答1"}},
  {{"Question": "新的提问2", "Reply": "新的回答2"}}
]
"""


async def augment_row(row, max_retries=3):
    sub_label = row['sub_label']
    n_variants = AUGMENT_MULTIPLIERS.get(sub_label, 0)
    if n_variants == 0:
        return []

    # 兼容 CSV 里可能是 Qsubj 或者是 Question 的情况
    q_text = row.get('Question', row.get('Qsubj', ''))

    prompt = PROMPT_TEMPLATE.format(
        sub_label=sub_label,
        n_variants=n_variants,
        question=q_text,
        reply=row['Reply']
    )

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
            )

            # ================= 核心修复区：增强代理网关兼容性 =================
            if isinstance(response, str):
                try:
                    resp_dict = json.loads(response)
                    result_str = resp_dict["choices"][0]["message"]["content"]
                except:
                    raise ValueError(f"网关返回了非标准字符串: {response[:100]}...")
            else:
                result_str = (response.choices[0].message.content or "")
            # ==================================================================

            result_str = result_str.strip()

            # 清理 Markdown 代码块
            if result_str.startswith("```"):
                result_str = result_str.strip("`").strip()
                if result_str.lower().startswith("json"):
                    result_str = result_str[4:].strip()

            variants = json.loads(result_str)

            synthetic_rows = []
            for v in variants:
                synthetic_rows.append({
                    # 强行统一为 Question，方便后续模型训练调用
                    "Question": v.get("Question", v.get("Qsubj", "")),
                    "Reply": v.get("Reply", ""),
                    "label": row['label'],
                    "sub_label": sub_label,
                    "is_synthetic": True
                })
            return synthetic_rows


        except Exception as e:

            print(f"\n[尝试 {attempt + 1}/{max_retries} 失败] 标签:{sub_label} | 错误: {type(e).__name__} - {str(e)}")
            # 【新增：错误显影液】如果是 JSON 解析错误，把罪魁祸首打印出来！
            if isinstance(e, json.JSONDecodeError) and  'result_str' in locals():
                print(f"❌ 惹祸的 LLM 原始输出:\n{result_str}\n" + "-" * 50)

            await asyncio.sleep(2)

    return []


async def process_chunk(chunk_rows, pbar):
    tasks = [augment_row(row) for row in chunk_rows]
    results = await asyncio.gather(*tasks)

    flattened_results = []
    for res in results:
        flattened_results.extend(res)

    pbar.update(len(chunk_rows))
    return flattened_results


async def main_async():
    df = pd.read_csv(INPUT_FILE)

    # 筛选出需要扩充的少数类
    df_target = df[df['sub_label'].isin(TARGET_CLASSES)].copy()
    print(f"找到需要扩充的少数类样本共 {len(df_target)} 条。")

    # 将 DataFrame 转换为字典列表方便处理
    rows = df_target.to_dict('records')
    chunks = [rows[i:i + CONCURRENCY_LIMIT] for i in range(0, len(rows), CONCURRENCY_LIMIT)]

    all_synthetic_data = []

    with tqdm(total=len(rows), desc="LLM 数据合成中") as pbar:
        for chunk in chunks:
            chunk_results = await process_chunk(chunk, pbar)
            all_synthetic_data.extend(chunk_results)
            await asyncio.sleep(0.5)

    df_synthetic = pd.DataFrame(all_synthetic_data)
    df_synthetic.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')

    print(f"\n✅ 数据增强完成！成功生成 {len(df_synthetic)} 条高质量合成样本。")
    print(f"合成数据已保存至: {OUTPUT_FILE}")
    print("\n=== 新增合成样本分布 ===")
    print(df_synthetic['sub_label'].value_counts())


if __name__ == "__main__":
    asyncio.run(main_async())