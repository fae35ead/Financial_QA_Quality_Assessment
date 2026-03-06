import pandas as pd
import json
import asyncio
import os
from openai import AsyncOpenAI
from openai import AuthenticationError, APIError, BadRequestError, RateLimitError
from tqdm.asyncio import tqdm

# ================= 配置区 =================
INPUT_FILE = "../data/processed/labeled_golden_samples.csv"
OUTPUT_FILE = "../data/processed/labeled_subnodes_samples.csv"
CHECKPOINT_FILE = "../data/processed/labeled_subnodes_checkpoint.csv"

API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "")
CONCURRENCY_LIMIT = 5

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= Prompt 定义 (LCPPN 子节点专供) =================

# 1. Direct 节点 (0) 子分类 Prompt
PROMPT_DIRECT = """你现在是一位资深的 A 股市场证券分析师。
你的任务是对已经被判定为【Direct (直接响应)】的董秘回答，进行深度的【业务领域分类】。
请仔细阅读给定的【投资者提问】和【董秘回答】，将其归入以下 5 个核心分类中的【唯一一个】：

1. "财务表现指引"：涉及利润、营收、毛利率的具体预期或财务数据。
2. "产能与项目规划"：涉及工厂建设、产能爬坡、固定资产投资、订单数量。
3. "技术与研发进展"：涉及专利、新产品流片、临床试验进度、技术突破。
4. "资本运作与并购"：涉及增发、重组、股权激励、分红、股份回购、股东人数。
5. "合规与风险披露"：涉及诉讼、环保处罚、退市风险提示、监管问询。

=== 输出格式 ===
请仅输出一个 JSON 对象，绝不能包含任何 Markdown 符号（如 ```json）。
【极其重要】：reason 字段内部引用请务必使用【单引号】（' '）。
示例：
{
  "sub_label": "产能与项目规划",
  "reason": "提问涉及二期工程，回答直接给出了'预计三季度投产'的具体节点"
}
"""

# 2. Evasive 节点 (2) 子分类 Prompt (基于 EvasionBench)
PROMPT_EVASIVE = """你现在是一位精通金融语用学与行为金融学的行为分析专家。
你的任务是对已经被判定为【Evasive (完全逃避/打太极)】的董秘回答，进行深度的【逃避战术 (Evasion Tactics) 鉴定】。
请仔细阅读给定的【投资者提问】和【董秘回答】，将其归入以下 4 个 EvasionBench 权威分类中的【唯一一个】：

1. "转移话题"：不回答核心数字或进展，而是大谈公司宏观战略、行业利好或国家政策。
2. "战略性模糊"：使用'预计'、'可能'、'不排除'等强对冲词，给出极其宽泛的区间，看似回答实则没有任何承诺。
3. "外部归因"：面对业绩下滑或项目延期，将责任完全推给疫情、地缘政治、宏观经济周期等外部因素。
4. "推迟回答"：在没有明确时间表的情况下，纯粹用'请关注后续定期报告'、'如有进展将及时披露'等合规话术挡箭牌。

=== 输出格式 ===
请仅输出一个 JSON 对象，绝不能包含任何 Markdown 符号（如 ```json）。
【极其重要】：reason 字段内部引用请务必使用【单引号】（' '）。
示例：
{
  "sub_label": "外部归因",
  "reason": "回答避开了自身经营问题，将营收下滑归咎于'全球宏观经济周期下行'"
}
"""


async def get_sublabel_async(question: str, reply: str, root_label: int, max_retries=3) -> dict:
    """根据根节点(root_label)的判定结果，动态选择 Prompt 进行二次打标"""
    # 过滤掉 Intermediate (1)，如果是 1，直接返回默认值
    if root_label == 1:
        return {"sub_label": "部分响应(不细分)", "sub_reason": "根节点为1，无需细分"}

    system_prompt = PROMPT_DIRECT if root_label == 0 else PROMPT_EVASIVE
    user_prompt = f"提问：{question}\n回答：{reply}"
    last_error = "UNKNOWN_ERROR"

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": f"{system_prompt}\n\n==========\n\n{user_prompt}"}],
                temperature=0.1,
            )

            # ================= 核心修复区：增强代理网关兼容性 =================
            if isinstance(response, str):
                try:
                    # 尝试将代理返回的 JSON 字符串解析为字典
                    resp_dict = json.loads(response)
                    result_str = resp_dict["choices"][0]["message"]["content"]
                except:
                    # 如果连 JSON 都不是，说明是网关报错网页文本
                    raise ValueError(f"网关返回了非标准字符串: {response[:100]}...")
            else:
                # 正常情况：标准的 OpenAI 对象
                result_str = (response.choices[0].message.content or "")
            # ==================================================================

            result_str = result_str.strip()

            # 清洗 Markdown 符号
            if result_str.startswith("```"):
                result_str = result_str.strip("`").strip()
                if result_str.lower().startswith("json"):
                    result_str = result_str[4:].strip()

            result_json = json.loads(result_str)

            if "sub_label" in result_json and "reason" in result_json:
                return {"sub_label": result_json["sub_label"], "sub_reason": result_json["reason"]}
            raise ValueError(f"JSON 缺少必要字段: {result_str[:200]}")

        except Exception as e:
            last_error = f"ERROR: {type(e).__name__}: {str(e)}"
            # 打印出具体错误，方便我们在控制台监控
            print(f"\n[尝试 {attempt + 1}/{max_retries} 失败] {last_error}")
            await asyncio.sleep(2)

    return {"sub_label": "打标失败", "sub_reason": last_error[:500]}


async def process_chunk(df, indices, pbar):
    tasks = [get_sublabel_async(df.loc[i, 'Qsubj'], df.loc[i, 'Reply'], df.loc[i, 'label']) for i in indices]
    results = await asyncio.gather(*tasks)

    for i, res in zip(indices, results):
        df.at[i, 'sub_label'] = res['sub_label']
        df.at[i, 'sub_reason'] = res['sub_reason']
        pbar.update(1)


async def main_async():
    print(f"正在加载根节点金标准数据: {INPUT_FILE}")

    if os.path.exists(CHECKPOINT_FILE):
        print(f"发现中断备份文件，从 {CHECKPOINT_FILE} 恢复进度...")
        df = pd.read_csv(CHECKPOINT_FILE)
    else:
        df = pd.read_csv(INPUT_FILE)

        df['sub_label'] = "待打标"
        df['sub_reason'] = ""

    # 我们只需要对 label 为 0 或 2，且还没打成功的数据进行二次打标
    # label 为 1 的在 get_sublabel_async 会直接被秒处理
    pending_indices = df[df['sub_label'] == "待打标"].index.tolist()
    print(f"共有 {len(pending_indices)} 条数据需要子节点微标注。")

    chunks = [pending_indices[i:i + CONCURRENCY_LIMIT] for i in range(0, len(pending_indices), CONCURRENCY_LIMIT)]

    with tqdm(total=len(pending_indices), desc="LCPPN 子节点打标进度") as pbar:
        for chunk in chunks:
            await process_chunk(df, chunk, pbar)
            df.to_csv(CHECKPOINT_FILE, index=False, encoding='utf-8-sig')
            await asyncio.sleep(1.0)  # 呼吸控制，防限流

    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    print(f"\n✅ LCPPN 数据打标完成！已保存至: {OUTPUT_FILE}")

    print("\n=== 子节点分类统计 (Direct 分支) ===")
    print(df[df['label'] == 0]['sub_label'].value_counts())

    print("\n=== 子节点分类统计 (Evasive 分支) ===")
    print(df[df['label'] == 2]['sub_label'].value_counts())


if __name__ == "__main__":
    asyncio.run(main_async())