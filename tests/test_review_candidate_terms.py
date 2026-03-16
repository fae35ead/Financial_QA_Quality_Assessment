import csv
import importlib.util
import sys
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "utils" / "review_candidate_terms.py"
MODULE_NAME = "review_candidate_terms"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
review_module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[MODULE_NAME] = review_module
SPEC.loader.exec_module(review_module)


def write_candidate_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["term", "tf", "df", "tfidf_score"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_load_candidates_requires_term_column(tmp_path: Path):
    bad_tsv = tmp_path / "bad.tsv"
    bad_tsv.write_text("foo\tbar\nx\ty\n", encoding="utf-8")
    with pytest.raises(ValueError):
        review_module.load_candidates(bad_tsv)


def test_review_candidates_retries_on_invalid_input():
    candidates = [{"term": "土地", "tf": "10", "df": "5", "tfidf_score": "3.2"}]
    decisions = iter(["invalid", "y"])
    messages: list[str] = []

    def fake_input(_: str) -> str:
        return next(decisions)

    result = review_module.review_candidates(candidates=candidates, input_fn=fake_input, output_fn=messages.append)
    assert result.entity_terms == ["土地"]
    assert result.stopword_terms == []
    assert result.ignored_terms == []
    assert any("无效输入" in msg for msg in messages)


def test_run_review_dedup_and_incremental_merge(tmp_path: Path):
    candidate_tsv = tmp_path / "candidate.tsv"
    write_candidate_tsv(
        candidate_tsv,
        rows=[
            {"term": "土地", "tf": "10", "df": "5", "tfidf_score": "3.2"},
            {"term": "公告", "tf": "9", "df": "4", "tfidf_score": "2.1"},
            {"term": "土地", "tf": "8", "df": "3", "tfidf_score": "1.9"},
            {"term": "并购", "tf": "7", "df": "2", "tfidf_score": "4.5"},
        ],
    )

    reviewed_entities = tmp_path / "custom_entities.reviewed.txt"
    reviewed_stopwords = tmp_path / "custom_stopwords.reviewed.txt"
    entities_dict = tmp_path / "custom_entities.txt"
    stopwords_dict = tmp_path / "custom_stopwords.txt"
    entities_dict.write_text("土地\n", encoding="utf-8")
    stopwords_dict.write_text("公告\n", encoding="utf-8")

    prompts: list[str] = []
    decisions = iter(["y", "s", "y"])  # 仅会对唯一词条提问：土地、公告、并购

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return next(decisions)

    summary = review_module.run_review(
        input_tsv=candidate_tsv,
        reviewed_entities_path=reviewed_entities,
        reviewed_stopwords_path=reviewed_stopwords,
        entities_dict_path=entities_dict,
        stopwords_dict_path=stopwords_dict,
        input_fn=fake_input,
        output_fn=lambda _: None,
    )

    assert len(prompts) == 3  # 重复词条“土地”不再二次询问
    assert summary["entity_terms"] == ["土地", "并购"]
    assert summary["stopword_terms"] == ["公告"]
    assert summary["appended_entities"] == ["并购"]
    assert summary["appended_stopwords"] == []

    assert reviewed_entities.read_text(encoding="utf-8").splitlines() == ["土地", "并购"]
    assert reviewed_stopwords.read_text(encoding="utf-8").splitlines() == ["公告"]

    assert entities_dict.read_text(encoding="utf-8").splitlines() == ["土地", "并购"]
    assert stopwords_dict.read_text(encoding="utf-8").splitlines() == ["公告"]
