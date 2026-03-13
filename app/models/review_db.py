'''阶段C持久化数据模型：定义问答样本、模型输出、复核记录、任务与审计表。'''

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


def _uuid() -> str:
    return str(uuid4())


class QASample(Base):
    __tablename__ = "qa_samples"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    company_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    qa_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    raw_source: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_qa_samples_company_time", "company_name", "qa_time"),
    )


class ModelOutput(Base):
    __tablename__ = "model_outputs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    sample_id: Mapped[str] = mapped_column(String(36), ForeignKey("qa_samples.id"), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False, default="2.1.0")
    layer1_label: Mapped[str] = mapped_column(String(128), nullable=False)
    layer1_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    layer2_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    is_low_confidence: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    review_status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    processed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("idx_model_outputs_sample", "sample_id"),
        Index("idx_model_outputs_low_conf", "is_low_confidence"),
        Index("idx_model_outputs_review_status", "review_status"),
    )


class Annotation(Base):
    __tablename__ = "annotations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    sample_id: Mapped[str] = mapped_column(String(36), ForeignKey("qa_samples.id"), nullable=False)
    source: Mapped[str] = mapped_column(String(16), nullable=False)  # model | agent | human
    annotator_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    annotation: Mapped[dict] = mapped_column(JSON, nullable=False)
    annotator_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    audit_trail: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    __table_args__ = (
        Index("idx_annotations_sample", "sample_id"),
        Index("idx_annotations_source", "source"),
    )


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    job_type: Mapped[str] = mapped_column(String(32), nullable=False)
    params: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        Index("idx_jobs_status", "status"),
    )


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(36), nullable=False)
    user_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    old_value: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    new_value: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
