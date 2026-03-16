'''推理接口路由：提供单条分析接口与历史兼容接口。'''

from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import InferRequest, InferResponse

router = APIRouter(tags=["inference"])


# 统一封装推理流程与异常映射，减少重复代码
def _evaluate(request: Request, payload: InferRequest) -> InferResponse:
    service = request.app.state.inference_service  # 从应用上下文获取共享推理服务
    review_service = request.app.state.review_service
    try:
        result = service.evaluate(payload.question, payload.answer)
        # 阶段C副作用：推理结果落库并执行低置信度入队判定。
        persisted = review_service.record_inference(
            payload={
                "company_name": payload.company_name,
                "qa_time": payload.qa_time,
                "question": payload.question,
                "answer": payload.answer,
                "sample_id": None,
            },
            result=result,
        )
        return InferResponse(
            **result,
            sample_id=persisted["sample_id"],
            is_low_confidence=persisted["is_low_confidence"],
            review_status=persisted["review_status"],
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc  # 入参问题
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc  # 运行依赖或模型问题
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"推理服务异常: {exc}") from exc  # 未预期异常


# 新版单条推理接口
@router.post("/infer", response_model=InferResponse)
def infer(payload: InferRequest, request: Request):
    return _evaluate(request, payload)


# 兼容旧前端调用路径，避免页面改造前接口断裂
@router.post("/api/evaluate", response_model=InferResponse)
def evaluate_compat(payload: InferRequest, request: Request):
    return _evaluate(request, payload)
