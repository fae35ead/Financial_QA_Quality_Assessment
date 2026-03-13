'''该文件实现了对 A 股董秘问答数据的自动化 LLM 打标，核心功能包括：
1. 读取待标注的 CSV 数据，支持断点续标和结果修复。
2. 定义了一个详细的 Prompt，指导大模型根据金融逃避战术（Evasion Tactics）对董秘回答的质量进行评估，并输出 JSON 格式的标签、置信度和理由。
3. 使用 AsyncOpenAI 客户端异步调用大模型接口，支持多次重试和详细的错误分类（鉴权错误、请求错误、限流错误、API 错误、解析错误等），并将错误原因透传到结果中，方便后续分析。
4. 实现了基于 asyncio 的并发处理逻辑，将待标注数据分块（Chunking）后同时发送多个请求，大幅提升处理效率，同时使用 tqdm 显示异步进度条。
5. 在每块数据处理完成后立即保存 checkpoint，确保即使中途发生意外也能最大程度保留已完成的工作，避免重复劳动。
6. 最终将标注结果保存到指定的输出文件，并提供一个简单的质量报告，包括标签分布和低置信度数据的统计，帮助用户快速了解标注结果的整体情况和潜在问题。'''

import pandas as pd
import json
import asyncio
import os
from pathlib import Path
from openai import AsyncOpenAI
from openai import AuthenticationError, APIError, BadRequestError, RateLimitError
from tqdm.asyncio import tqdm

# ================= 配置区 =================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "golden_samples_to_label.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "labeled_golden_samples.csv"
CHECKPOINT_FILE = PROJECT_ROOT / "data" / "processed" / "labeled_golden_samples_checkpoint.csv"

# 优先从环境变量读取，避免把密钥硬编码进代码
API_KEY = os.getenv("LLM_API_KEY", "")
BASE_URL = os.getenv("LLM_BASE_URL", "")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "")

# 【关键配置】并发数量 (Concurrency Limit)
CONCURRENCY_LIMIT = 5


def validate_config():
    """启动前检查关键配置，避免跑满一轮后才发现是配置错误。"""
    if not API_KEY.strip():
        raise ValueError("未检测到 LLM_API_KEY。请先在环境变量中配置 API Key。")
    if not BASE_URL.strip():
        raise ValueError("未检测到 LLM_BASE_URL。请配置正确的服务地址。")
    if not MODEL_NAME.strip():
        raise ValueError("未检测到 LLM_MODEL_NAME。请配置正确的模型名称。")

    print(f"当前模型: {MODEL_NAME}")
    print(f"当前 BASE_URL: {BASE_URL}")

    if "agentrouter.org" in BASE_URL:
        print("提示: 你当前走的是 AgentRouter 网关，必须使用该平台签发的有效 Key。")
    elif "bigmodel.cn" in BASE_URL:
        print("提示: 你当前走的是智谱官方兼容接口，必须使用智谱平台的 API Key。")


validate_config()
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# ================= Prompt 定义 =================
SYSTEM_PROMPT = """你现在是一位资深的 A 股市场证券分析师及 NLP 数据标注专家。
你的任务是评估上市公司董秘在互动易平台上对投资者提问的【回答质量】（即回答的逃避程度 Evasiveness）。

请仔细阅读给定的【投资者提问】和【董秘回答】，并严格按照以下基于“金融逃避战术（Evasion Tactics）”的标注标准，将回答分类为 0、1、2 三个等级，并以 JSON 格式输出理由和标签。

=== 核心原则 (CORE PRINCIPLE) ===
1. 寻找“核心诉求(Primary Ask)”：忽略投资者的情绪宣泄，定位其真正索要的客观事实、数据或进展。
2. 评判标准：根据回答是否满足了“核心诉求”来打分，而不是根据回答的字数长短或语气是否礼貌。

=== 标签定义与 A 股边界规则 ===

【0: Direct (直接响应)】
- 给出了具体的量化数据或明确的定性结论。
- A 股特殊规则 [合规性断言]：面对敏感或违规提问，极其干脆地拒绝（如：“不属于披露范围”、“公司暂无此业务”），不拖泥带水，也记为 Direct。
- [提供增量]：给出了具体的财务数字、时间节点或精确的业务状态（如：产能、订单量、具体股东人数）。

【1: Intermediate (部分响应)】
- 避重就轻：提问包含多个点，董秘只挑了最容易或最不敏感的问题回答，对核心的尖锐问题避而不谈。
- 定性代替定量：投资者要具体数字，董秘只给模糊的方向（如“稳步增长”、“符合预期”）。

【2: Fully Evasive (完全逃避/打太极)】
核心：触发了以下任何一种 EvasionBench 逃避战术：
1. [转移话题 (Topic Shifting)]：用大量篇幅谈论宏观经济、国家政策、行业趋势，但只字不提提问中涉及的具体项目。
2. [同义复述 (Redundant Repetition)]：把投资者提问里的词汇重新组合说了一遍，完全没有提供任何新信息。
3. [推迟回答 (Deferred Answer)]：在没有明确时间表的情况下，使用“请关注后续定期报告”、“将适时披露”来推脱（注：如果投资者问的就是财报数据，此类回答算标签 0 的合规性断言；如果问的是具体业务进展，用财报来挡箭牌则算标签 2）。
4. [虚假安抚 (Fake Reassurance)]：“感谢您的建议，我们将转达管理层”、“您的意见极其宝贵”等纯粹的 PR 话术。

=== 输出格式 ===
请仅输出一个 JSON 对象，绝不能包含任何 Markdown 符号（如 ```json），直接输出合法的字典结构。
【极其重要】：在 reason 字段中，如果需要引用文本，请务必使用【单引号】（' '），绝对禁止使用双引号（" "），否则会导致 JSON 解析失败！
示例：
{
  "label": 0,
  "confidence": 0.95,
  "reason": "此处填写50字以内理由，内部引用如'稳步推进'请用单引号"
}
"""


async def get_llm_label_async(question: str, reply: str, max_retries=3) -> dict:
    """异步调用大模型，并将具体错误原因透传到结果里。"""
    user_prompt = f"提问：{question}\n回答：{reply}"
    last_error = "UNKNOWN_ERROR"

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": f"{SYSTEM_PROMPT}\n\n==========\n\n{user_prompt}"
                    }
                ],
                temperature=0.1,
                # response_format={"type": "json_object"}
            )

            if isinstance(response, str):
                raise ValueError(f"网关发生异常，返回了纯文本网页而不是标准对象。内容摘要: {response[:200]}")
            result_str = (response.choices[0].message.content or "").strip()

            if result_str.startswith("```"):
                result_str = result_str.strip("`").strip()
                if result_str.lower().startswith("json"):
                    result_str = result_str[4:].strip()

            result_json = json.loads(result_str)

            if "label" in result_json and "confidence" in result_json and "reason" in result_json:
                return result_json
            raise ValueError(f"JSON 缺少必要字段，模型返回内容为: {result_str[:200]}")

        except AuthenticationError as e:
            last_error = f"AUTH_401: {str(e)}"
            print(f"\n[尝试 {attempt + 1}/{max_retries} 失败] 鉴权错误: {e}")
            break
        except BadRequestError as e:
            last_error = f"BAD_REQUEST: {str(e)}"
            print(f"\n[尝试 {attempt + 1}/{max_retries} 失败] 请求参数错误: {e}")
            # 常见于 response_format / model / messages 不被当前网关支持，继续重试通常没意义
            break
        except RateLimitError as e:
            last_error = f"RATE_LIMIT: {str(e)}"
            print(f"\n[尝试 {attempt + 1}/{max_retries} 失败] 限流错误: {e}")
            await asyncio.sleep(3)
        except (APIError, json.JSONDecodeError, ValueError) as e:
            last_error = f"API_OR_PARSE_ERROR: {type(e).__name__}: {str(e)}"
            print(f"\n[尝试 {attempt + 1}/{max_retries} 失败] 错误类型: {type(e).__name__} | 详情: {e}")
            await asyncio.sleep(2)
        except Exception as e:
            last_error = f"UNEXPECTED_ERROR: {type(e).__name__}: {str(e)}"
            print(f"\n[尝试 {attempt + 1}/{max_retries} 失败] 未知错误类型: {type(e).__name__} | 详情: {e}")
            await asyncio.sleep(2)

    return {"label": -1, "confidence": 0.0, "reason": last_error[:500]}


async def process_chunk(df, indices, pbar):
    """处理一批数据：并发发送所有请求，收集结果后统一更新 DataFrame"""
    # 创建这一批次的所有的任务
    tasks = [get_llm_label_async(df.loc[i, 'Qsubj'], df.loc[i, 'Reply']) for i in indices]

    # 瞬间把这十几条任务同时发出去，并等待它们全部归来
    results = await asyncio.gather(*tasks)

    # 统一把结果写回 DataFrame（Pandas 是同步库，这样写最安全）
    for i, res in zip(indices, results):
        df.at[i, 'label'] = res['label']
        df.at[i, 'confidence'] = res['confidence']
        df.at[i, 'reason'] = res['reason']
        pbar.update(1)


async def main_async():
    print(f"正在加载待打标数据: {INPUT_FILE}")

    if os.path.exists(OUTPUT_FILE):
        print(f"发现已存在的最终文件，从 {OUTPUT_FILE} 加载，准备修复失败的漏网之鱼...")
        df = pd.read_csv(OUTPUT_FILE)
    elif os.path.exists(CHECKPOINT_FILE):
        print(f"发现中断备份文件，从 {CHECKPOINT_FILE} 恢复进度...")
        df = pd.read_csv(CHECKPOINT_FILE)
    else:
        print(f"首次运行，从 {INPUT_FILE} 加载原始数据...")
        df = pd.read_csv(INPUT_FILE)
        df['label'] = -1
        df['confidence'] = 0.0
        df['reason'] = ""

    pending_indices = df[df['label'] == -1].index.tolist()
    print(f"共有 {len(pending_indices)} 条数据需要打标。")
    print(f"当前设置并发量为: {CONCURRENCY_LIMIT} 条/次")

    # 【核心逻辑】将剩下的任务按并发量“切块”(Chunking)
    chunks = [pending_indices[i:i + CONCURRENCY_LIMIT] for i in range(0, len(pending_indices), CONCURRENCY_LIMIT)]

    # 启动异步进度条
    with tqdm(total=len(pending_indices), desc="LLM 异步打标进度") as pbar:
        for chunk in chunks:
            # 异步处理完一整块 (比如 15 条)
            await process_chunk(df, chunk, pbar)
            # 一块处理完，立刻保存一次 checkpoint
            df.to_csv(CHECKPOINT_FILE, index=False, encoding='utf-8-sig')

    # 收尾工作
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    print(f"\n✅ 所有数据打标完成！已保存至: {OUTPUT_FILE}")

    print("\n=== 数据标注质量报告 ===")
    print(df['label'].value_counts().sort_index())
    low_conf_count = len(df[(df['confidence'] < 0.6) & (df['label'] != -1)])
    print(f"置信度低于 0.6 的边缘数据共计: {low_conf_count} 条，建议人工抽检。")
    print("\n失败原因 Top10:")
    print(df[df['label'] == -1]['reason'].value_counts().head(10))


if __name__ == "__main__":
    asyncio.run(main_async())
