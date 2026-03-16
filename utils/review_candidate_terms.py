"""交互式候选词审核脚本：支持 y/n/s 审核并回流到词表文件。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


InputFn = Callable[[str], str]
OutputFn = Callable[[str], None]


# 解析词典行：兼容“词\t词频”与“纯词”，仅保留词本体
def parse_dictionary_keyword(raw_line: str) -> str:
    normalized = raw_line.strip()
    if not normalized:
        return ""
    keyword = normalized.split()[0]
    return keyword if len(keyword) > 1 else ""


# 读取现有词表并转换为集合，便于去重
def load_existing_terms(dict_path: Path) -> set[str]:
    terms: set[str] = set()
    if not dict_path.exists():
        return terms
    with dict_path.open("r", encoding="utf-8") as f:
        for line in f:
            keyword = parse_dictionary_keyword(line)
            if keyword:
                terms.add(keyword)
    return terms


# 加载候选词 TSV，要求存在 term 列；同词仅保留首条记录
def load_candidates(tsv_path: Path) -> list[dict[str, str]]:
    with tsv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        fields = reader.fieldnames or []
        if "term" not in fields:
            raise ValueError("候选词文件缺少必需列: term")

        rows: list[dict[str, str]] = []
        seen_terms: set[str] = set()
        for row in reader:
            term = (row.get("term") or "").strip()
            if len(term) <= 1:
                continue
            if term in seen_terms:
                continue
            seen_terms.add(term)
            rows.append(
                {
                    "term": term,
                    "tf": (row.get("tf") or "").strip(),
                    "df": (row.get("df") or "").strip(),
                    "tfidf_score": (row.get("tfidf_score") or "").strip(),
                }
            )
    return rows


# 为单条候选词构造可读展示文本
def format_candidate_line(idx: int, total: int, row: dict[str, str]) -> str:
    return (
        f"[{idx}/{total}] term={row['term']} | tf={row.get('tf', '')} | "
        f"df={row.get('df', '')} | tfidf={row.get('tfidf_score', '')}"
    )


# 单条输入校验：仅接受 y / n / s
def ask_decision(input_fn: InputFn, output_fn: OutputFn) -> str:
    while True:
        decision = input_fn("请输入决策 [y=加入实体词典, n=忽略, s=加入停用词]: ").strip().lower()
        if decision in {"y", "n", "s"}:
            return decision
        output_fn("无效输入，请输入 y / n / s。")


@dataclass
class ReviewResult:
    entity_terms: list[str]
    stopword_terms: list[str]
    ignored_terms: list[str]


# 交互式审核：逐条展示并采集用户决策
def review_candidates(
    candidates: list[dict[str, str]],
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> ReviewResult:
    entity_terms: list[str] = []
    stopword_terms: list[str] = []
    ignored_terms: list[str] = []
    total = len(candidates)

    for idx, row in enumerate(candidates, start=1):
        output_fn(format_candidate_line(idx, total, row))
        decision = ask_decision(input_fn=input_fn, output_fn=output_fn)
        term = row["term"]
        if decision == "y":
            entity_terms.append(term)
        elif decision == "s":
            stopword_terms.append(term)
        else:
            ignored_terms.append(term)
    return ReviewResult(entity_terms=entity_terms, stopword_terms=stopword_terms, ignored_terms=ignored_terms)


# 重写输出 reviewed 文件（本次审核快照）
def write_reviewed_terms(output_path: Path, terms: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for term in terms:
            f.write(f"{term}\n")


# 判断追加写入时是否需要先补一个换行
def needs_leading_newline(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as f:
        f.seek(-1, 2)
        return f.read(1) != b"\n"


# 只把“新增词”追加到正式词表，保持幂等
def append_new_terms(dict_path: Path, reviewed_terms: list[str]) -> list[str]:
    existing_terms = load_existing_terms(dict_path)
    append_terms = [term for term in reviewed_terms if term not in existing_terms]
    if not append_terms:
        return []

    dict_path.parent.mkdir(parents=True, exist_ok=True)
    prefix_newline = needs_leading_newline(dict_path)
    with dict_path.open("a", encoding="utf-8", newline="\n") as f:
        if prefix_newline:
            f.write("\n")
        for term in append_terms:
            f.write(f"{term}\n")
    return append_terms


# 执行完整审核流程：加载 -> 交互 -> reviewed 写入 -> 正式词表增量回流
def run_review(
    input_tsv: Path,
    reviewed_entities_path: Path,
    reviewed_stopwords_path: Path,
    entities_dict_path: Path,
    stopwords_dict_path: Path,
    limit: int | None = None,
    input_fn: InputFn = input,
    output_fn: OutputFn = print,
) -> dict[str, object]:
    candidates = load_candidates(input_tsv)
    if limit is not None:
        candidates = candidates[:limit]
    output_fn(f"待审核候选词数量: {len(candidates)}")

    review_result = review_candidates(candidates=candidates, input_fn=input_fn, output_fn=output_fn)
    write_reviewed_terms(reviewed_entities_path, review_result.entity_terms)
    write_reviewed_terms(reviewed_stopwords_path, review_result.stopword_terms)

    appended_entities = append_new_terms(entities_dict_path, review_result.entity_terms)
    appended_stopwords = append_new_terms(stopwords_dict_path, review_result.stopword_terms)

    summary = {
        "total": len(candidates),
        "entity_terms": review_result.entity_terms,
        "stopword_terms": review_result.stopword_terms,
        "ignored_terms": review_result.ignored_terms,
        "appended_entities": appended_entities,
        "appended_stopwords": appended_stopwords,
        "reviewed_entities_path": str(reviewed_entities_path),
        "reviewed_stopwords_path": str(reviewed_stopwords_path),
        "entities_dict_path": str(entities_dict_path),
        "stopwords_dict_path": str(stopwords_dict_path),
    }
    return summary


# 解析 CLI 参数，支持相对项目根目录路径
def resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = project_root / raw_path
    return path.resolve()


# 脚本入口
def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="交互式审核候选词并回流词表。")
    parser.add_argument("--input", default="data/others/candidate_entities.auto.tsv", help="候选词 TSV 文件。")
    parser.add_argument(
        "--entities-reviewed",
        default="data/others/custom_entities.reviewed.txt",
        help="本次审核实体词快照输出文件。",
    )
    parser.add_argument(
        "--stopwords-reviewed",
        default="data/others/custom_stopwords.reviewed.txt",
        help="本次审核停用词快照输出文件。",
    )
    parser.add_argument(
        "--entities-dict",
        default="data/others/custom_entities.txt",
        help="正式实体词典（仅增量追加）。",
    )
    parser.add_argument(
        "--stopwords-dict",
        default="data/others/custom_stopwords.txt",
        help="正式停用词词典（仅增量追加，不存在则创建）。",
    )
    parser.add_argument("--limit", type=int, default=None, help="仅审核前 N 条候选词（调试可用）。")
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        raise ValueError("参数 --limit 必须为正整数。")

    summary = run_review(
        input_tsv=resolve_path(project_root, args.input),
        reviewed_entities_path=resolve_path(project_root, args.entities_reviewed),
        reviewed_stopwords_path=resolve_path(project_root, args.stopwords_reviewed),
        entities_dict_path=resolve_path(project_root, args.entities_dict),
        stopwords_dict_path=resolve_path(project_root, args.stopwords_dict),
        limit=args.limit,
    )

    print("--- 审核完成 ---")
    print(f"总候选词: {summary['total']}")
    print(f"加入实体词: {len(summary['entity_terms'])} (新增写入 {len(summary['appended_entities'])})")
    print(f"加入停用词: {len(summary['stopword_terms'])} (新增写入 {len(summary['appended_stopwords'])})")
    print(f"忽略词条: {len(summary['ignored_terms'])}")
    print(f"实体快照: {summary['reviewed_entities_path']}")
    print(f"停用词快照: {summary['reviewed_stopwords_path']}")


if __name__ == "__main__":
    main()
