'''推理服务层：封装模型加载、实体门卫与两层分类推理逻辑。'''

from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from app.core.config import Settings


# 解析词典行：兼容“词\t词频”与“纯词”两种格式，仅保留关键词本体
def parse_dictionary_keyword(raw_line: str) -> str:
    normalized = raw_line.strip()
    if not normalized:
        return ""
    keyword = normalized.split()[0]  # 按空白切分，自动去掉词频列
    return keyword if len(keyword) > 1 else ""


# 单条问答推理服务：负责模型生命周期管理和推理执行
class InferenceService:
    ROOT_LABELS = {0: "Direct (直接响应)", 1: "Intermediate (避重就轻)", 2: "Evasive (打太极)"}
    DIRECT_LABELS = {
        0: "资本运作与并购",
        1: "技术与研发进展",
        2: "产能与项目规划",
        3: "合规与风险披露",
        4: "财务表现指引",
    }
    EVASIVE_LABELS = {0: "推迟回答", 1: "转移话题", 2: "战略性模糊", 3: "外部归因"}

    # 初始化服务状态与运行时依赖占位符
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = Lock()  # 避免并发初始化导致重复加载模型
        self._initialized = False

        self._torch = None
        self._nnf = None
        self._tokenizer_cls = None
        self._model_cls = None
        self._keyword_processor_cls = None

        self.device = None
        self.tokenizer = None
        self.root_model = None
        self.direct_model = None
        self.evasive_model = None
        self.keyword_processor = None

    # 只读属性：供外部判断模型是否已经就绪
    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # 延迟导入重依赖，降低应用启动压力并改善缺依赖提示
    def _ensure_runtime_modules(self) -> None:
        if self._torch is not None:
            return

        try:
            import torch
            import torch.nn.functional as nnf
            from flashtext import KeywordProcessor
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "运行依赖缺失，请先安装 requirements.txt（至少包括 torch/transformers/flashtext）。"
            ) from exc

        self._torch = torch  # 深度学习运行时
        self._nnf = nnf
        self._keyword_processor_cls = KeywordProcessor
        self._model_cls = AutoModelForSequenceClassification
        self._tokenizer_cls = AutoTokenizer

    # 加载单个分类模型并切换到推理模式
    def _load_model(self, model_path: Path):
        if not model_path.exists():
            raise RuntimeError(f"模型加载失败，找不到路径: {model_path}")
        model = self._model_cls.from_pretrained(str(model_path))  # 从本地权重目录加载
        model.to(self.device)  # 挂载到 CPU/GPU
        model.eval()  # 关闭 Dropout，确保推理稳定
        return model

    # 将概率向量映射为“标签 -> 概率(0~1)”字典，便于前端直接渲染图表
    @staticmethod
    def _to_probability_map(probabilities: list[float], label_map: dict[int, str]) -> dict[str, float]:
        return {
            label_map.get(index, str(index)): round(float(prob), 6)
            for index, prob in enumerate(probabilities)
        }

    # 从单段文本提取实体命中和字符位置，供前端做高亮
    def _extract_entity_hits(self, text: str, source_text: str) -> list[dict[str, Any]]:
        hits = self.keyword_processor.extract_keywords(text, span_info=True)
        return [
            {
                "text": keyword,
                "start": int(start),
                "end": int(end),
                "source_text": source_text,
            }
            for keyword, start, end in hits
        ]

    # 构建金融实体词典门卫，用于拦截非金融或低质量样本
    def _build_keyword_processor(self):
        keyword_processor = self._keyword_processor_cls()
        dict_files = [
            self.settings.project_root / "data" / "others" / "THUOCL_caijing.txt",
            self.settings.project_root / "data" / "others" / "baostock_entities.txt",
            self.settings.project_root / "data" / "others" / "custom_entities.txt",
        ]

        for file_path in dict_files:
            if not file_path.exists():
                continue
            with file_path.open("r", encoding="utf-8") as f:
                words = [parse_dictionary_keyword(line) for line in f]
                words = [word for word in words if word]
                keyword_processor.add_keywords_from_list(words)  # 批量装载词典词条
        return keyword_processor

    # 初始化推理上下文（线程安全，重复调用无副作用）
    def initialize(self) -> None:
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            self._ensure_runtime_modules()

            self.device = self._torch.device("cuda" if self._torch.cuda.is_available() else "cpu")  # 自动选择设备
            self.keyword_processor = self._build_keyword_processor()
            self.tokenizer = self._tokenizer_cls.from_pretrained(str(self.settings.root_model_dir))
            self.root_model = self._load_model(self.settings.root_model_dir)
            self.direct_model = self._load_model(self.settings.direct_model_dir)
            self.evasive_model = self._load_model(self.settings.evasive_model_dir)
            self._initialized = True

    # 通用节点推理：输出预测标签、置信度与完整概率向量
    def _predict_node(self, model: Any, input_ids: Any, attention_mask: Any) -> tuple[int, float, list[float]]:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = self._nnf.softmax(outputs.logits, dim=-1).squeeze(0)
        confidence, pred_id = self._torch.max(probs, dim=-1)
        return pred_id.item(), confidence.item(), probs.detach().cpu().tolist()

    # 对单条问答执行端到端两层推理
    def evaluate(self, question: str, answer: str) -> dict[str, Any]:
        question = question.strip()
        answer = answer.strip()

        if not question or not answer:
            raise ValueError("提问和回答不能为空")

        self.initialize()  # 懒加载触发点：首次推理时加载模型

        combined_text = question + answer
        question_hits = self._extract_entity_hits(question, source_text="question")
        answer_hits = self._extract_entity_hits(answer, source_text="answer")
        entity_hits = question_hits + answer_hits
        # 前置门卫：长度过短或无金融实体时直接拦截
        if len(combined_text) < self.settings.entity_min_length or len(entity_hits) == 0:
            return {
                "root_id": -1,
                "root_label": "[拦截] 非金融领域对话",
                "root_confidence": 100.0,
                "sub_label": "无关噪音或闲聊",
                "sub_confidence": 100.0,
                "root_probabilities": {},
                "sub_probabilities": {},
                "entity_hits": entity_hits,
                "warning": "实体门卫触发：未检测到有效金融实体，已拦截该样本。",
            }

        # 交叉编码器输入格式：question + answer
        encoded = self.tokenizer(
            text=question,
            text_pair=answer,
            truncation=True,
            max_length=self.settings.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)  # 文本token张量
        attention_mask = encoded["attention_mask"].to(self.device)  # 注意力掩码

        warning_msg = None
        sub_probabilities: dict[str, float] = {}
        # 关闭梯度，减少显存和计算开销
        with self._torch.no_grad():
            root_id, root_conf, root_probs = self._predict_node(self.root_model, input_ids, attention_mask)
            if root_conf < self.settings.low_conf_threshold:
                warning_msg = "根节点置信度过低，该问答可能超出常规金融语境，建议人工复核。"

            # LCPPN 路由：按根节点结果进入不同子分类器
            if root_id == 0:
                sub_id, sub_conf, sub_probs = self._predict_node(self.direct_model, input_ids, attention_mask)
                sub_label = self.DIRECT_LABELS.get(sub_id, "未知")
                sub_probabilities = self._to_probability_map(sub_probs, self.DIRECT_LABELS)
            elif root_id == 2:
                sub_id, sub_conf, sub_probs = self._predict_node(self.evasive_model, input_ids, attention_mask)
                sub_label = self.EVASIVE_LABELS.get(sub_id, "未知")
                sub_probabilities = self._to_probability_map(sub_probs, self.EVASIVE_LABELS)
            else:
                sub_label = "无下游细分 (部分响应)"
                sub_conf = 1.0

        return {
            "root_id": root_id,
            "root_label": self.ROOT_LABELS.get(root_id, "未知"),
            "root_confidence": round(root_conf * 100, 2),
            "sub_label": sub_label,
            "sub_confidence": round(sub_conf * 100, 2),
            "root_probabilities": self._to_probability_map(root_probs, self.ROOT_LABELS),
            "sub_probabilities": sub_probabilities,
            "entity_hits": entity_hits,
            "warning": warning_msg,
        }
