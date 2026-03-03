from contextlib import asynccontextmanager

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer


# TODO: 请将下面路径替换为你本地训练好的模型绝对路径
MODEL_PATH = "/ABSOLUTE/PATH/TO/YOUR/BERT_QA_QUALITY_MODEL"

LABEL_MAP = {
    0: "低质量",
    1: "中等质量",
    2: "高质量",
}


class EvaluateRequest(BaseModel):
    question: str = Field(..., min_length=1, description="股东提问内容")
    answer: str = Field(..., min_length=1, description="AI生成的草稿回答")


class EvaluateResponse(BaseModel):
    label_id: int
    label_name: str
    confidence: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 在应用启动时加载 tokenizer 与模型，避免重复加载
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    except Exception as e:
        raise RuntimeError(
            "模型加载失败，请检查 MODEL_PATH 是否替换为正确的模型绝对路径"
        ) from e

    model.to(device)
    model.eval()

    app.state.device = device
    app.state.tokenizer = tokenizer
    app.state.model = model

    yield


app = FastAPI(title="QA质量评估服务", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health_check():
    return {"status": "ok", "device": str(app.state.device)}


@app.post("/api/evaluate", response_model=EvaluateResponse)
def evaluate(payload: EvaluateRequest):
    question = payload.question.strip()
    answer = payload.answer.strip()

    if not question or not answer:
        raise HTTPException(status_code=400, detail="question 和 answer 不能为空")

    # 按要求使用 [SEP] 拼接输入
    merged_text = f"{question} [SEP] {answer}"

    encoded = app.state.tokenizer(
        merged_text,
        truncation=True,
        max_length=512,
        padding=True,
        return_tensors="pt",
    )
    encoded = {k: v.to(app.state.device) for k, v in encoded.items()}

    with torch.no_grad():
        outputs = app.state.model(**encoded)
        probs = F.softmax(outputs.logits, dim=-1)[0]

    label_id = int(torch.argmax(probs).item())
    confidence = round(float(probs[label_id].item() * 100), 2)

    return EvaluateResponse(
        label_id=label_id,
        label_name=LABEL_MAP.get(label_id, "未知"),
        confidence=confidence,
    )
