'''API 数据模型：定义请求与响应的结构化契约。'''

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# 单条推理请求模型
class InferRequest(BaseModel):
    company_name: Optional[str] = Field(default=None, description="公司名称，可选")
    qa_time: Optional[datetime] = Field(default=None, description="问答时间，可选")
    question: str = Field(..., min_length=1, description="投资者提问内容")
    answer: str = Field(..., min_length=1, description="董秘回答内容")


# 单条推理响应模型
class EntityHit(BaseModel):
    text: str
    start: int
    end: int
    source_text: Literal["question", "answer"]


# 单条推理响应模型
class InferResponse(BaseModel):
    root_id: int
    root_label: str
    root_confidence: float
    sub_label: str
    sub_confidence: float
    root_probabilities: dict[str, float] = Field(default_factory=dict)
    sub_probabilities: dict[str, float] = Field(default_factory=dict)
    entity_hits: list[EntityHit] = Field(default_factory=list)
    warning: Optional[str] = None
    sample_id: Optional[str] = None
    is_low_confidence: Optional[bool] = None
    review_status: Optional[str] = None


# 批量推理中的单条样本结构
class BatchInferItem(BaseModel):
    sample_id: Optional[str] = Field(default=None, description="外部样本ID，可选")
    company_name: Optional[str] = Field(default=None, description="公司名称，可选")
    qa_time: Optional[datetime] = Field(default=None, description="问答时间，可选")
    question: str = Field(..., min_length=1, description="投资者提问内容")
    answer: str = Field(..., min_length=1, description="董秘回答内容")


# 批量推理请求模型
class BatchInferRequest(BaseModel):
    items: list[BatchInferItem] = Field(..., min_length=1, description="待批量分析样本列表")


# 批量推理创建响应模型
class BatchInferResponse(BaseModel):
    job_id: str
    status: str
    total: int


# 批任务中单条结果模型
class JobResultItem(BaseModel):
    index: int
    sample_id: Optional[str] = None
    status: str
    result: Optional[InferResponse] = None
    error: Optional[str] = None


# 批任务状态响应模型
class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    total: int
    completed: int
    failed: int
    progress: float
    created_at: str
    finished_at: Optional[str] = None
    results: list[JobResultItem] = Field(default_factory=list)


# 待复核队列单条记录
class ReviewQueueItem(BaseModel):
    sample_id: str
    company_name: Optional[str] = None
    qa_time: Optional[datetime] = None
    question_text: str
    answer_text: str
    layer1_label: str
    layer1_confidence: float
    layer2_json: dict[str, Any]
    review_status: Optional[str] = None
    is_low_confidence: bool
    processed_at: datetime


# 待复核队列分页响应
class ReviewQueueResponse(BaseModel):
    page: int
    page_size: int
    total: int
    items: list[ReviewQueueItem]


# 复核详情响应
class ReviewDetailResponse(BaseModel):
    sample_id: str
    company_name: Optional[str] = None
    qa_time: Optional[datetime] = None
    question_text: str
    answer_text: str
    model_output: dict[str, Any]
    agent_suggestion: Optional[dict[str, Any]] = None
    human_annotation: Optional[dict[str, Any]] = None


# Agent建议任务触发响应
class AgentSuggestionJobResponse(BaseModel):
    job_id: str
    status: str
    sample_id: str


# 手动加入待复核队列响应
class ManualReviewEnqueueResponse(BaseModel):
    sample_id: str
    review_status: str
    enqueued: bool


# 人工复核提交请求
class AnnotateRequest(BaseModel):
    sample_id: str = Field(..., min_length=1)
    root_label: str = Field(..., min_length=1)
    sub_label: str = Field(..., min_length=1)
    note: Optional[str] = None
    annotator_id: Optional[str] = None
    annotator_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


# 人工复核提交响应
class AnnotateResponse(BaseModel):
    sample_id: str
    review_status: str
    annotation_id: str
    training_corpus_file: str
