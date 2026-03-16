"""基于现有问答语料自动发现候选金融词条，供人工审核后补充词典。"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


# 抽取连续中文片段，后续在片段内做字符 n-gram
CHUNK_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}")
MULTI_SPACE_PATTERN = re.compile(r"\s+")
DEFAULT_EXTRA_STOP_TERMS = {
    "公司",
    "公司的",
    "贵公司",
    "对公司",
    "投资者",
    "谢谢",
    "您好",
    "请问",
    "相关",
    "是否",
    "目前",
    "关注",
    "公告",
}
COMMON_NOISE_PHRASES = [
    "感谢您对公司的关注",
    "感谢您的关注",
    "感谢您对公司的支持",
    "尊敬的投资者您好",
    "尊敬的投资者",
    "尊敬的股东您好",
    "尊敬的股东",
    "请关注公司后续公告",
    "请关注后续公告",
    "请关注公司公告",
    "请以公司公告为准",
    "敬请谅解",
    "谢谢",
    "您好",
]


# 解析词典行：兼容“词\t词频”与“纯词”格式
def parse_dictionary_keyword(raw_line: str) -> str:
    normalized = raw_line.strip()
    if not normalized:
        return ""
    keyword = normalized.split()[0]
    return keyword if len(keyword) > 1 else ""


# 加载现有词典集合，用于排除已存在词条
def load_existing_terms(dict_paths: Iterable[Path]) -> set[str]:
    terms: set[str] = set()
    for path in dict_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                keyword = parse_dictionary_keyword(line)
                if keyword:
                    terms.add(keyword)
    return terms


# 加载停用词集合（含内置常见套话词）
def load_stopwords(stopword_paths: Iterable[Path], extra_terms: Iterable[str]) -> set[str]:
    stopwords = {term.strip() for term in extra_terms if term.strip()}
    for path in stopword_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                term = line.strip()
                if len(term) > 1:
                    stopwords.add(term)
    return stopwords


# 从一段文本中提取中文字符 n-gram
def extract_cjk_ngrams(text: str, min_n: int, max_n: int) -> list[str]:
    grams: list[str] = []
    for chunk in CHUNK_PATTERN.findall(text):
        chunk_len = len(chunk)
        if chunk_len < min_n:
            continue
        local_max_n = min(max_n, chunk_len)
        for size in range(min_n, local_max_n + 1):
            for start in range(0, chunk_len - size + 1):
                grams.append(chunk[start : start + size])
    return grams


# 规则过滤：剔除明显功能词/语气词噪声
def is_valid_candidate(term: str, stopwords: set[str]) -> bool:
    if term in stopwords:
        return False
    if len(term) < 2:
        return False
    if len(term) <= 3 and "的" in term:
        return False
    if term.startswith(("的", "了")) or term.endswith(("的", "了", "吗", "呢", "吧", "呀", "啊")):
        return False
    return True


# 轻量去噪：剔除高频套话，减少候选词噪声
def normalize_text(raw_text: str) -> str:
    text = raw_text
    for phrase in COMMON_NOISE_PHRASES:
        text = text.replace(phrase, " ")
    return MULTI_SPACE_PATTERN.sub(" ", text).strip()


# 读取语料指定列并拼接成单条文本
def iter_text_rows(csv_path: Path, fields: list[str], max_rows: int | None) -> Iterable[str]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            if max_rows is not None and idx >= max_rows:
                break
            parts = [(row.get(field) or "").strip() for field in fields]
            merged = " ".join(part for part in parts if part)
            merged = normalize_text(merged)
            if merged:
                yield merged


# 核心统计：计算 TF、DF 和 TF-IDF 分数
def discover_candidates(
    rows: Iterable[str],
    min_n: int,
    max_n: int,
    min_count: int,
    top_k: int,
    existing_terms: set[str],
    stopwords: set[str] | None = None,
) -> tuple[list[dict[str, float | int | str]], int]:
    tf_counter: Counter[str] = Counter()
    df_counter: Counter[str] = Counter()
    doc_count = 0

    for text in rows:
        doc_count += 1
        grams = extract_cjk_ngrams(text, min_n=min_n, max_n=max_n)
        if not grams:
            continue
        tf_counter.update(grams)
        df_counter.update(set(grams))

    records: list[dict[str, float | int | str]] = []
    stopwords = stopwords or set()
    for term, tf in tf_counter.items():
        if tf < min_count:
            continue
        if term in existing_terms:
            continue
        if not is_valid_candidate(term, stopwords):
            continue
        df = int(df_counter[term])
        score = tf * math.log((1.0 + doc_count) / (1.0 + df))  # 轻量 TF-IDF 近似
        records.append(
            {
                "term": term,
                "tf": int(tf),
                "df": df,
                "tfidf_score": round(score, 6),
            }
        )

    records.sort(key=lambda item: (float(item["tfidf_score"]), int(item["tf"])), reverse=True)
    return records[:top_k], doc_count


# 导出候选词条明细，供人工审核
def write_candidates(output_path: Path, candidates: list[dict[str, float | int | str]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["term", "tf", "df", "tfidf_score"]
    with output_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(candidates)


# 路径参数解析：支持逗号分隔的绝对/相对路径
def resolve_paths(project_root: Path, raw_paths: str) -> list[Path]:
    paths: list[Path] = []
    for item in raw_paths.split(","):
        raw = item.strip()
        if not raw:
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = project_root / raw
        paths.append(p.resolve())
    return paths


# 脚本入口：读取参数并执行候选词发现
def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="从问答语料自动发现候选词条（需人工审核后入库）。")
    parser.add_argument(
        "--input",
        default=str(project_root / "data" / "processed" / "cleaned_data.csv"),
        help="输入语料 CSV 路径。",
    )
    parser.add_argument(
        "--output",
        default=str(project_root / "data" / "others" / "candidate_entities.auto.tsv"),
        help="候选词导出路径（TSV）。",
    )
    parser.add_argument("--fields", default="Qsubj,Reply", help="参与抽词的列名，逗号分隔。")
    parser.add_argument("--max-rows", type=int, default=None, help="最大扫描行数（默认全量）。")
    parser.add_argument("--min-n", type=int, default=2, help="最小 n-gram 长度。")
    parser.add_argument("--max-n", type=int, default=4, help="最大 n-gram 长度。")
    parser.add_argument("--min-count", type=int, default=20, help="最小词频阈值。")
    parser.add_argument("--top-k", type=int, default=500, help="输出候选词数量。")
    parser.add_argument(
        "--dicts",
        default="data/others/THUOCL_caijing.txt,data/others/baostock_entities.txt,data/others/custom_entities.txt",
        help="已有词典路径列表（相对项目根目录或绝对路径），逗号分隔。",
    )
    parser.add_argument(
        "--stopwords",
        default=(
            "data/others/停用词词表/cn_stopwords.txt,"
            "data/others/停用词词表/hit_stopwords.txt,"
            "data/others/custom_stopwords.txt"
        ),
        help="停用词路径列表（相对项目根目录或绝对路径），逗号分隔。",
    )
    parser.add_argument(
        "--extra-stop-terms",
        default=",".join(sorted(DEFAULT_EXTRA_STOP_TERMS)),
        help="额外停用词列表，逗号分隔。",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    fields = [item.strip() for item in args.fields.split(",") if item.strip()]
    if not fields:
        raise ValueError("参数 --fields 不能为空。")
    if args.min_n < 2 or args.max_n < args.min_n:
        raise ValueError("请确保 2 <= min_n <= max_n。")
    if args.min_count < 1 or args.top_k < 1:
        raise ValueError("min_count 和 top_k 必须为正整数。")

    dict_paths = resolve_paths(project_root, args.dicts)
    stopword_paths = resolve_paths(project_root, args.stopwords)
    extra_stop_terms = [item.strip() for item in args.extra_stop_terms.split(",") if item.strip()]

    existing_terms = load_existing_terms(dict_paths)
    stopwords = load_stopwords(stopword_paths, extra_stop_terms)
    candidates, doc_count = discover_candidates(
        rows=iter_text_rows(input_path, fields=fields, max_rows=args.max_rows),
        min_n=args.min_n,
        max_n=args.max_n,
        min_count=args.min_count,
        top_k=args.top_k,
        existing_terms=existing_terms,
        stopwords=stopwords,
    )
    write_candidates(output_path, candidates)

    print(f"[candidate-terms] docs={doc_count}, existing_terms={len(existing_terms)}, output={output_path}")
    print(
        f"[candidate-terms] candidates={len(candidates)} (top_k={args.top_k}, min_count={args.min_count}, stopwords={len(stopwords)})"
    )
    preview_size = min(20, len(candidates))
    for idx in range(preview_size):
        row = candidates[idx]
        print(
            f"{idx + 1:02d}. {row['term']} | tf={row['tf']} | df={row['df']} | tfidf={row['tfidf_score']}"
        )


if __name__ == "__main__":
    main()
