'''复核服务层：负责低置信度入队、复核详情、人工标注提交与训练集回流。'''

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.review_db import Annotation, AuditLog, Job, ModelOutput, QASample


class ReviewService:
    REVIEW_PENDING = "pending_review"
    REVIEW_AGENT_SUGGESTED = "agent_suggested"
    REVIEW_CONFIRMED = "confirmed"
    REVIEW_REVISED = "revised"
    REVIEW_STATUSES = {
        REVIEW_PENDING,
        REVIEW_AGENT_SUGGESTED,
        REVIEW_CONFIRMED,
        REVIEW_REVISED,
    }

    ROOT_LABELS = {
        "Direct (直接响应)",
        "Intermediate (避重就轻)",
        "Evasive (打太极)",
    }
    DIRECT_SUB_LABELS = {
        "资本运作与并购",
        "技术与研发进展",
        "产能与项目规划",
        "合规与风险披露",
        "财务表现指引",
    }
    EVASIVE_SUB_LABELS = {"推迟回答", "转移话题", "战略性模糊", "外部归因"}

    def __init__(self, settings):
        self.settings = settings
        self.sub_conf_threshold = float(os.getenv("QA_LOW_CONF_SUB_THRESHOLD", "45.0"))
        raw_longtail = os.getenv("QA_LONGTAIL_LABELS", "战略性模糊,外部归因,合规与风险披露").strip()
        self.longtail_labels = {x.strip() for x in raw_longtail.split(",") if x.strip()}
        default_corpus = settings.project_root / "data" / "processed" / "review_training_corpus.csv"
        raw_file = os.getenv("QA_TRAINING_CORPUS_FILE", str(default_corpus))
        self.training_corpus_file = Path(raw_file)
        if not self.training_corpus_file.is_absolute():
            self.training_corpus_file = (settings.project_root / self.training_corpus_file).resolve()

    def _root_threshold_percent(self) -> float:
        # 配置里是0~1概率阈值，推理输出是百分比。
        threshold = float(getattr(self.settings, "low_conf_threshold", 0.45))
        return threshold * 100 if threshold <= 1 else threshold

    def _is_low_confidence(self, result: dict[str, Any]) -> bool:
        root_conf = float(result.get("root_confidence", 0.0))
        sub_conf = float(result.get("sub_confidence", 0.0))
        sub_label = str(result.get("sub_label", "")).strip()
        return (
            root_conf < self._root_threshold_percent()
            or sub_conf < self.sub_conf_threshold
            or sub_label in self.longtail_labels
        )

    def _latest_model_output(self, db: Session, sample_id: str) -> ModelOutput | None:
        stmt = (
            select(ModelOutput)
            .where(ModelOutput.sample_id == sample_id)
            .order_by(desc(ModelOutput.processed_at))
            .limit(1)
        )
        return db.scalar(stmt)

    def _latest_annotation(self, db: Session, sample_id: str, source: str) -> Annotation | None:
        stmt = (
            select(Annotation)
            .where(Annotation.sample_id == sample_id, Annotation.source == source)
            .order_by(desc(Annotation.created_at))
            .limit(1)
        )
        return db.scalar(stmt)

    def _add_audit_log(
        self,
        db: Session,
        entity_type: str,
        entity_id: str,
        action: str,
        user_id: str | None,
        old_value: dict[str, Any] | None,
        new_value: dict[str, Any] | None,
    ) -> None:
        db.add(
            AuditLog(
                entity_type=entity_type,
                entity_id=entity_id,
                user_id=user_id,
                action=action,
                old_value=old_value,
                new_value=new_value,
            )
        )

    def record_inference(self, payload: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
        low_conf = self._is_low_confidence(result)
        review_status = self.REVIEW_PENDING if low_conf else None

        with SessionLocal() as db:
            sample = QASample(
                company_name=payload.get("company_name"),
                qa_time=payload.get("qa_time"),
                question_text=payload["question"],
                answer_text=payload["answer"],
                raw_source={
                    "source": "api",
                    "sample_id": payload.get("sample_id"),
                },
            )
            db.add(sample)
            db.flush()

            model_output = ModelOutput(
                sample_id=sample.id,
                model_version=str(getattr(self.settings, "app_version", "2.1.0")),
                layer1_label=result["root_label"],
                layer1_confidence=float(result["root_confidence"]),
                layer2_json={
                    "sub_label": result["sub_label"],
                    "sub_confidence": float(result["sub_confidence"]),
                    "warning": result.get("warning"),
                },
                is_low_confidence=low_conf,
                review_status=review_status,
            )
            db.add(model_output)
            db.flush()

            db.add(
                Annotation(
                    sample_id=sample.id,
                    source="model",
                    annotator_id=None,
                    annotation={
                        "root_label": result["root_label"],
                        "sub_label": result["sub_label"],
                        "warning": result.get("warning"),
                    },
                    annotator_confidence=float(result["root_confidence"]) / 100.0,
                    audit_trail={"phase": "inference"},
                )
            )
            if low_conf:
                self._add_audit_log(
                    db=db,
                    entity_type="model_output",
                    entity_id=model_output.id,
                    action="enqueue_review",
                    user_id=None,
                    old_value={"review_status": None},
                    new_value={"review_status": self.REVIEW_PENDING},
                )
            db.commit()

            return {
                "sample_id": sample.id,
                "model_output_id": model_output.id,
                "is_low_confidence": low_conf,
                "review_status": review_status,
            }

    def list_queue(
        self,
        page: int,
        page_size: int,
        statuses: list[str],
        date_from: datetime | None,
        date_to: datetime | None,
    ) -> dict[str, Any]:
        statuses = [s for s in statuses if s in self.REVIEW_STATUSES] or [
            self.REVIEW_PENDING,
            self.REVIEW_AGENT_SUGGESTED,
        ]

        with SessionLocal() as db:
            stmt = (
                select(QASample, ModelOutput)
                .join(ModelOutput, ModelOutput.sample_id == QASample.id)
                .where(ModelOutput.review_status.in_(statuses))
                .order_by(desc(ModelOutput.processed_at))
            )
            if date_from:
                stmt = stmt.where(ModelOutput.processed_at >= date_from)
            if date_to:
                stmt = stmt.where(ModelOutput.processed_at <= date_to)

            rows = db.execute(stmt).all()
            total = len(rows)
            start = (page - 1) * page_size
            end = start + page_size
            items = []
            for sample, output in rows[start:end]:
                items.append(
                    {
                        "sample_id": sample.id,
                        "company_name": sample.company_name,
                        "qa_time": sample.qa_time,
                        "question_text": sample.question_text,
                        "answer_text": sample.answer_text,
                        "layer1_label": output.layer1_label,
                        "layer1_confidence": output.layer1_confidence,
                        "layer2_json": output.layer2_json,
                        "review_status": output.review_status,
                        "is_low_confidence": output.is_low_confidence,
                        "processed_at": output.processed_at,
                    }
                )
            return {"total": total, "items": items}

    def get_review_detail(self, sample_id: str) -> dict[str, Any]:
        with SessionLocal() as db:
            sample = db.get(QASample, sample_id)
            if not sample:
                raise ValueError(f"样本不存在: {sample_id}")

            model_output = self._latest_model_output(db, sample_id)
            if not model_output:
                raise ValueError(f"样本缺少模型输出: {sample_id}")
            agent_annotation = self._latest_annotation(db, sample_id, "agent")
            human_annotation = self._latest_annotation(db, sample_id, "human")

            return {
                "sample_id": sample.id,
                "company_name": sample.company_name,
                "qa_time": sample.qa_time,
                "question_text": sample.question_text,
                "answer_text": sample.answer_text,
                "model_output": {
                    "layer1_label": model_output.layer1_label,
                    "layer1_confidence": model_output.layer1_confidence,
                    "layer2_json": model_output.layer2_json,
                    "is_low_confidence": model_output.is_low_confidence,
                    "review_status": model_output.review_status,
                    "processed_at": model_output.processed_at,
                },
                "agent_suggestion": agent_annotation.annotation if agent_annotation else None,
                "human_annotation": human_annotation.annotation if human_annotation else None,
            }

    def create_agent_job(self, sample_id: str, requester_id: str | None = None) -> str:
        with SessionLocal() as db:
            sample = db.get(QASample, sample_id)
            if not sample:
                raise ValueError(f"样本不存在: {sample_id}")
            model_output = self._latest_model_output(db, sample_id)
            if not model_output:
                raise ValueError(f"样本缺少模型输出: {sample_id}")
            if model_output.review_status not in {self.REVIEW_PENDING, self.REVIEW_AGENT_SUGGESTED}:
                raise ValueError("仅待复核样本允许请求 Agent 建议。")

            job = Job(
                job_type="agent_suggestion",
                params={"sample_id": sample_id, "requester_id": requester_id},
                status="pending",
                progress=0.0,
            )
            db.add(job)
            db.flush()
            self._add_audit_log(
                db=db,
                entity_type="job",
                entity_id=job.id,
                action="create_agent_suggestion_job",
                user_id=requester_id,
                old_value=None,
                new_value={"sample_id": sample_id},
            )
            db.commit()
            return job.id

    def mark_job_started(self, job_id: str) -> None:
        with SessionLocal() as db:
            job = db.get(Job, job_id)
            if not job:
                return
            job.status = "running"
            job.progress = 10.0
            db.commit()

    def finish_agent_job(self, job_id: str, sample_id: str, suggestion: dict[str, Any]) -> None:
        with SessionLocal() as db:
            job = db.get(Job, job_id)
            if not job:
                raise ValueError(f"任务不存在: {job_id}")
            sample = db.get(QASample, sample_id)
            if not sample:
                raise ValueError(f"样本不存在: {sample_id}")
            model_output = self._latest_model_output(db, sample_id)
            if not model_output:
                raise ValueError(f"样本缺少模型输出: {sample_id}")

            model_output.review_status = self.REVIEW_AGENT_SUGGESTED
            db.add(
                Annotation(
                    sample_id=sample_id,
                    source="agent",
                    annotator_id="dify",
                    annotation=suggestion,
                    annotator_confidence=float(suggestion.get("confidence", 0.0)),
                    audit_trail={"job_id": job_id},
                )
            )
            job.status = "completed"
            job.progress = 100.0
            job.finished_at = datetime.utcnow()
            self._add_audit_log(
                db=db,
                entity_type="model_output",
                entity_id=model_output.id,
                action="agent_suggestion_created",
                user_id="dify",
                old_value={"review_status": self.REVIEW_PENDING},
                new_value={"review_status": self.REVIEW_AGENT_SUGGESTED},
            )
            db.commit()

    def fail_job(self, job_id: str, message: str) -> None:
        with SessionLocal() as db:
            job = db.get(Job, job_id)
            if not job:
                return
            job.status = "failed"
            job.progress = 100.0
            job.error_message = message
            job.finished_at = datetime.utcnow()
            db.commit()

    def get_persisted_job(self, job_id: str) -> dict[str, Any] | None:
        with SessionLocal() as db:
            job = db.get(Job, job_id)
            if not job:
                return None
            return {
                "job_id": job.id,
                "status": job.status,
                "progress": job.progress,
                "created_at": job.created_at.isoformat(),
                "finished_at": job.finished_at.isoformat() if job.finished_at else None,
                "params": job.params,
                "error": job.error_message,
            }

    def _validate_human_annotation(self, root_label: str, sub_label: str) -> None:
        if root_label not in self.ROOT_LABELS:
            raise ValueError("根标签非法。")
        if root_label.startswith("Direct") and sub_label not in self.DIRECT_SUB_LABELS:
            raise ValueError("Direct 根标签下的子标签非法。")
        if root_label.startswith("Evasive") and sub_label not in self.EVASIVE_SUB_LABELS:
            raise ValueError("Evasive 根标签下的子标签非法。")
        if root_label.startswith("Intermediate") and sub_label != "无下游细分 (部分响应)":
            raise ValueError("Intermediate 根标签下子标签必须为“无下游细分 (部分响应)”。")

    def _write_training_corpus(
        self,
        sample: QASample,
        final_root_label: str,
        final_sub_label: str,
        annotator_confidence: float | None,
        note: str | None,
    ) -> None:
        self.training_corpus_file.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "sample_id",
            "company_name",
            "qa_time",
            "question_text",
            "answer_text",
            "root_label",
            "sub_label",
            "annotator_confidence",
            "note",
            "created_at",
        ]
        row = {
            "sample_id": sample.id,
            "company_name": sample.company_name or "",
            "qa_time": sample.qa_time.isoformat() if sample.qa_time else "",
            "question_text": sample.question_text,
            "answer_text": sample.answer_text,
            "root_label": final_root_label,
            "sub_label": final_sub_label,
            "annotator_confidence": annotator_confidence if annotator_confidence is not None else "",
            "note": note or "",
            "created_at": datetime.utcnow().isoformat(),
        }

        existing_rows: list[dict[str, Any]] = []
        if self.training_corpus_file.exists():
            with self.training_corpus_file.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)

        replaced = False
        for idx, item in enumerate(existing_rows):
            if item.get("sample_id") == sample.id:
                existing_rows[idx] = row
                replaced = True
                break
        if not replaced:
            existing_rows.append(row)

        with self.training_corpus_file.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(existing_rows)

    def submit_human_annotation(
        self,
        sample_id: str,
        root_label: str,
        sub_label: str,
        note: str | None,
        annotator_id: str | None,
        annotator_confidence: float | None,
    ) -> dict[str, Any]:
        self._validate_human_annotation(root_label, sub_label)
        with SessionLocal() as db:
            sample = db.get(QASample, sample_id)
            if not sample:
                raise ValueError(f"样本不存在: {sample_id}")

            model_output = self._latest_model_output(db, sample_id)
            if not model_output:
                raise ValueError(f"样本缺少模型输出: {sample_id}")
            if model_output.review_status not in {self.REVIEW_PENDING, self.REVIEW_AGENT_SUGGESTED}:
                raise ValueError("当前样本状态不允许提交复核。")

            model_root = model_output.layer1_label
            model_sub = str(model_output.layer2_json.get("sub_label", ""))
            final_status = (
                self.REVIEW_CONFIRMED
                if model_root == root_label and model_sub == sub_label
                else self.REVIEW_REVISED
            )

            old_status = model_output.review_status
            model_output.review_status = final_status

            human_annotation = Annotation(
                sample_id=sample_id,
                source="human",
                annotator_id=annotator_id,
                annotation={
                    "root_label": root_label,
                    "sub_label": sub_label,
                    "note": note,
                },
                annotator_confidence=annotator_confidence,
                audit_trail={
                    "old_model_root_label": model_root,
                    "old_model_sub_label": model_sub,
                },
            )
            db.add(human_annotation)
            db.flush()

            self._add_audit_log(
                db=db,
                entity_type="model_output",
                entity_id=model_output.id,
                action="human_annotation_submit",
                user_id=annotator_id,
                old_value={"review_status": old_status},
                new_value={"review_status": final_status},
            )
            db.commit()

            self._write_training_corpus(
                sample=sample,
                final_root_label=root_label,
                final_sub_label=sub_label,
                annotator_confidence=annotator_confidence,
                note=note,
            )

            return {
                "sample_id": sample_id,
                "review_status": final_status,
                "annotation_id": human_annotation.id,
                "training_corpus_file": str(self.training_corpus_file),
            }
