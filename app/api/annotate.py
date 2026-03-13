'''人工复核接口路由：提交人工确认/修改结果并完成训练集回流。'''

from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import AnnotateRequest, AnnotateResponse

router = APIRouter(tags=["annotation"])


@router.post("/annotate", response_model=AnnotateResponse)
def annotate(payload: AnnotateRequest, request: Request):
    service = request.app.state.review_service
    try:
        result = service.submit_human_annotation(
            sample_id=payload.sample_id,
            root_label=payload.root_label,
            sub_label=payload.sub_label,
            note=payload.note,
            annotator_id=payload.annotator_id or "human_reviewer",
            annotator_confidence=payload.annotator_confidence,
        )
        return AnnotateResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
