import os
from contextlib import asynccontextmanager
from typing import Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ================= 生产环境配置区 =================
# TODO: 请务必将以下路径替换为你本地模型的绝对路径！
ROOT_MODEL_DIR = r"F:\软件\学习相关\PycharmProjects\QA\models\student_distilbert"
DIRECT_MODEL_DIR = r"F:\软件\学习相关\PycharmProjects\QA\models\lcppn_direct_classifier"
EVASIVE_MODEL_DIR = r"F:\软件\学习相关\PycharmProjects\QA\models\lcppn_evasive_classifier"

MAX_LEN = 384

# ================= 业务标签字典 =================
ROOT_LABELS = {0: "Direct (直接响应)", 1: "Intermediate (避重就轻)", 2: "Evasive (打太极)"}
DIRECT_LABELS = {0: "资本运作与并购", 1: "技术与研发进展", 2: "产能与项目规划", 3: "合规与风险披露", 4: "财务表现指引"}
EVASIVE_LABELS = {0: "推迟回答", 1: "转移话题", 2: "战略性模糊", 3: "外部归因"}


class EvaluateRequest(BaseModel):
    question: str = Field(..., min_length=1, description="投资者提问内容")
    answer: str = Field(..., min_length=1, description="董秘回答内容")


class EvaluateResponse(BaseModel):
    root_id: int
    root_label: str
    root_confidence: float
    sub_label: str
    sub_confidence: float
    warning: Optional[str] = None


def _load_model(model_path, device):
    if not os.path.exists(model_path):
        raise RuntimeError(f"模型加载失败，找不到路径: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("⚙️ [System] 正在初始化 LCPPN 联调推理引擎...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(ROOT_MODEL_DIR)
        root_model = _load_model(ROOT_MODEL_DIR, device)
        direct_model = _load_model(DIRECT_MODEL_DIR, device)
        evasive_model = _load_model(EVASIVE_MODEL_DIR, device)
    except Exception as e:
        raise RuntimeError("模型初始化异常，请检查绝对路径！") from e

    app.state.device = device
    app.state.tokenizer = tokenizer
    app.state.root_model = root_model
    app.state.direct_model = direct_model
    app.state.evasive_model = evasive_model
    print("✅ 三个模型已就绪！")
    yield


app = FastAPI(title="LCPPN 金融问答质量评估服务", version="2.0.0", lifespan=lifespan)

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

def _predict_node(model, input_ids, attention_mask):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    probs = F.softmax(outputs.logits, dim=-1)
    confidence, pred_id = torch.max(probs, dim=-1)
    return pred_id.item(), confidence.item()


@app.post("/api/evaluate", response_model=EvaluateResponse)
async def evaluate(payload: EvaluateRequest):
    question = payload.question.strip()
    answer = payload.answer.strip()

    if not question or not answer:
        raise HTTPException(status_code=400, detail="提问和回答不能为空")

    # 交叉编码器标准输入：text + text_pair
    encoded = app.state.tokenizer(
        text=question,
        text_pair=answer,
        truncation=True,
        max_length=MAX_LEN,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encoded['input_ids'].to(app.state.device)
    attention_mask = encoded['attention_mask'].to(app.state.device)

    warning_msg = None

    with torch.no_grad():
        # 1. 根节点推理
        root_id, root_conf = _predict_node(app.state.root_model, input_ids, attention_mask)

        if root_conf < 0.45:
            warning_msg = "根节点置信度过低，该问答可能超出常规金融语境，建议人工复核。"

        # 2. 路由到子节点
        if root_id == 0:
            sub_id, sub_conf = _predict_node(app.state.direct_model, input_ids, attention_mask)
            sub_label = DIRECT_LABELS.get(sub_id, "未知")
        elif root_id == 2:
            sub_id, sub_conf = _predict_node(app.state.evasive_model, input_ids, attention_mask)
            sub_label = EVASIVE_LABELS.get(sub_id, "未知")
        else:
            sub_label = "无下游细分 (部分响应)"
            sub_conf = 1.0

    return EvaluateResponse(
        root_id=root_id,
        root_label=ROOT_LABELS.get(root_id, "未知"),
        root_confidence=round(root_conf * 100, 2),
        sub_label=sub_label,
        sub_confidence=round(sub_conf * 100, 2),
        warning=warning_msg
    )