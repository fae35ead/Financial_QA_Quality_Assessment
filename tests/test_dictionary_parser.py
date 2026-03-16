'''该测试文件用于测试字典解析函数 parse_dictionary_keyword 的正确性。'''

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.services.inference_service import parse_dictionary_keyword


@pytest.mark.parametrize(
    "raw_line,expected",
    [
        ("房地产\t165034", "房地产"),
        ("净利润 9999", "净利润"),
        ("毛利率", "毛利率"),
        ("   ", ""),
        ("\n", ""),
        ("A", ""),
    ],
)
def test_parse_dictionary_keyword_supports_word_and_word_freq(raw_line, expected):
    assert parse_dictionary_keyword(raw_line) == expected
