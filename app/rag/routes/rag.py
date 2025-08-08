"""GPT-OSS RAG 시스템 API 엔드포인트."""

from fastapi import APIRouter, Depends, HTTPException, Query

from app.rag.representations.request import RAGRequest
from app.rag.representations.response import (
    HealthResponse,
    RAGQueryResponse,
    RAGResponse,
)
from app.rag.services import GPTOSSService, RAGService

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.get("/health/", response_model=HealthResponse)
async def check_rag_health(rag_service: RAGService = Depends(RAGService)):
    """RAG 시스템 상태 확인."""
    try:
        health_status = await rag_service.check_service_health()
        return HealthResponse(**health_status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"상태 확인 실패: {str(e)}")


@router.post("/answer/", response_model=RAGResponse)
async def generate_rag_answer(
    request: RAGRequest, rag_service: RAGService = Depends(RAGService)
):
    """RAG 답변 생성."""
    try:
        # RAG 답변 생성
        answer = await rag_service.generate_answer(
            question=request.question,
            context_documents=request.context_documents,
            user_id="test_user",
            temperature=request.temperature,
        )

        return RAGResponse(
            question=request.question,
            answer=answer,
            context_count=len(request.context_documents),
            model_info=rag_service.gpt_oss_service.get_model_info(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"답변 생성 실패: {str(e)}")


@router.post("/query/", response_model=RAGQueryResponse)
async def process_rag_query(
    question: str = Query(..., description="사용자 질문", min_length=1),
    user_id: str = Query("default", description="사용자 ID"),
    max_documents: int = Query(5, ge=1, le=10, description="최대 검색 문서 수"),
    similarity_threshold: float = Query(
        0.7, ge=0.0, le=1.0, description="유사도 임계값"
    ),
    temperature: float = Query(0.1, ge=0.0, le=1.0, description="답변 창의성"),
    rag_service: RAGService = Depends(RAGService),
):
    """Vector DB 기반 RAG 질의 처리."""
    try:
        result = await rag_service.process_rag_query(
            question=question,
            user_id=user_id,
            max_documents=max_documents,
            similarity_threshold=similarity_threshold,
            temperature=temperature,
        )

        return RAGQueryResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG 처리 실패: {str(e)}")


@router.get("/model/info/")
async def get_model_info(gpt_oss_service: GPTOSSService = Depends(GPTOSSService)):
    """GPT-OSS 모델 정보 조회."""
    try:
        return gpt_oss_service.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 정보 조회 실패: {str(e)}")


@router.post("/model/pull/")
async def pull_model(gpt_oss_service: GPTOSSService = Depends(GPTOSSService)):
    """GPT-OSS 모델 다운로드."""
    try:
        success = await gpt_oss_service.pull_model()

        if success:
            return {"message": "모델 다운로드 완료", "model": gpt_oss_service.model}
        else:
            raise HTTPException(status_code=500, detail="모델 다운로드 실패")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 다운로드 중 오류: {str(e)}")


# 샘플 테스트 데이터
@router.get("/sample/")
async def get_sample_test():
    """샘플 테스트 데이터 반환."""
    return {
        "sample_request": {
            "question": "FastAPI의 장점은 무엇인가요?",
            "context_documents": [
                "FastAPI는 Python으로 작성된 현대적이고 빠른 웹 프레임워크입니다. 자동 API 문서 생성, 타입 힌트 지원, 높은 성능을 제공합니다.",
                "FastAPI는 비동기 처리를 지원하여 높은 동시성을 제공합니다. 또한 Pydantic을 사용한 데이터 검증과 직렬화를 지원합니다.",
            ],
            "temperature": 0.1,
        },
        "usage": "POST /rag/answer/ 엔드포인트에 위 sample_request를 전송하세요",
    }
