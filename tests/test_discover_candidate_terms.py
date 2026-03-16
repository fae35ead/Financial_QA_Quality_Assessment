'''该测试文件是用于测试 discover_candidate_terms.py 模块的单元测试，确保其核心功能正常工作。'''

import csv
import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "discover_candidate_terms.py"
SPEC = importlib.util.spec_from_file_location("discover_candidate_terms", MODULE_PATH)
discover_module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(discover_module)


def test_parse_dictionary_keyword_supports_frequency_suffix():
    assert discover_module.parse_dictionary_keyword("房地产\t165034") == "房地产"
    assert discover_module.parse_dictionary_keyword("净利润 999") == "净利润"


def test_discover_candidates_excludes_existing_terms():
    rows = [
        "公司土地拍卖计划进展顺利",
        "土地拍卖事项会按计划推进",
        "拍卖相关计划仍在评估",
    ]
    candidates, doc_count = discover_module.discover_candidates(
        rows=rows,
        min_n=2,
        max_n=2,
        min_count=2,
        top_k=20,
        existing_terms={"土地"},
    )
    terms = {item["term"] for item in candidates}
    assert doc_count == 3
    assert "土地" not in terms
    assert "拍卖" in terms


def test_iter_text_rows_reads_selected_fields(tmp_path: Path):
    sample_csv = tmp_path / "sample.csv"
    with sample_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Qsubj", "Reply", "Other"])
        writer.writeheader()
        writer.writerow({"Qsubj": "问题A", "Reply": "回答A", "Other": "x"})
        writer.writerow({"Qsubj": "问题B", "Reply": "回答B", "Other": "y"})

    rows = list(discover_module.iter_text_rows(sample_csv, fields=["Qsubj", "Reply"], max_rows=None))
    assert rows == ["问题A 回答A", "问题B 回答B"]


def test_discover_candidates_applies_stopwords_filter():
    rows = ["公司利润增长明显", "公司利润持续增长", "利润增长趋势明确"]
    candidates, _ = discover_module.discover_candidates(
        rows=rows,
        min_n=2,
        max_n=2,
        min_count=2,
        top_k=20,
        existing_terms=set(),
        stopwords={"公司"},
    )
    terms = {item["term"] for item in candidates}
    assert "公司" not in terms
    assert "利润" in terms


def test_normalize_text_removes_common_boilerplate():
    text = "尊敬的投资者您好，感谢您对公司的关注，请关注后续公告。谢谢"
    normalized = discover_module.normalize_text(text)
    assert "投资者" not in normalized
    assert "关注后续公告" not in normalized
