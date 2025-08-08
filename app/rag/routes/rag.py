"""GPT-OSS RAG 시스템 API 엔드포인트."""

from fastapi import APIRouter, Depends, HTTPException

from app.rag.representations.request import RAGRequest, RAGQueryParametersRequest
from app.rag.representations.response import (
    HealthResponse,
    RAGQueryResponse,
    RAGResponse,
)
from app.rag.services import GPTOSSService, RAGService

router = APIRouter(prefix="/rag", tags=["RAG"])


@router.get(
    "/health/",
    response_model=HealthResponse,
)
async def check_rag_health(
    rag_service: RAGService = Depends(RAGService),
):
    """RAG 시스템 상태 확인."""
    try:
        health_status = await rag_service.check_service_health()
        return HealthResponse(**health_status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"상태 확인 실패: {str(e)}")


@router.post("/answer/", response_model=RAGResponse)
async def generate_rag_answer(
    payload: RAGRequest,
    rag_service: RAGService = Depends(RAGService),
):
    """RAG 답변 생성."""
    try:
        # RAG 답변 생성
        answer = await rag_service.generate_answer(
            question=payload.question,
            context_documents=payload.context_documents,
            user_id="test_user",
            temperature=payload.temperature,
        )

        return RAGResponse(
            question=payload.question,
            answer=answer,
            context_count=len(payload.context_documents),
            model_info=rag_service.gpt_oss_service.get_model_info(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"답변 생성 실패: {str(e)}")


@router.post("/query/", response_model=RAGQueryResponse)
async def process_rag_query(
    payload: RAGQueryParametersRequest,
    rag_service: RAGService = Depends(RAGService),
):
    """Vector DB 기반 RAG 질의 처리."""
    try:
        result = await rag_service.process_rag_query(
            question=payload.question,
            user_id=payload.user_id,
            max_documents=payload.max_documents,
            similarity_threshold=payload.similarity_threshold,
            temperature=payload.temperature,
        )

        return RAGQueryResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG 처리 실패: {str(e)}")


@router.get("/model/info/")
async def get_model_info(
    gpt_oss_service: GPTOSSService = Depends(GPTOSSService),
):
    """GPT-OSS 모델 정보 조회."""
    try:
        return gpt_oss_service.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 정보 조회 실패: {str(e)}")


@router.post("/model/pull/")
async def pull_model(
    gpt_oss_service: GPTOSSService = Depends(GPTOSSService),
):
    """GPT-OSS 모델 다운로드."""
    try:
        success = await gpt_oss_service.pull_model()

        if success:
            return {"message": "모델 다운로드 완료", "model": gpt_oss_service.model}
        else:
            raise HTTPException(status_code=500, detail="모델 다운로드 실패")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 다운로드 중 오류: {str(e)}")


@router.get("/database/status/")
async def check_database_status(
    rag_service: RAGService = Depends(RAGService),
):
    """벡터 데이터베이스 상태 확인."""
    try:
        db_status = await rag_service.vector_search_service.check_database_status()
        return {
            "database": "vector_database",
            "status": db_status,
            "recommendation": _get_db_recommendation(db_status),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"데이터베이스 상태 확인 실패: {str(e)}"
        )


def _get_db_recommendation(db_status: dict) -> str:
    """데이터베이스 상태에 따른 권장사항 반환."""
    if "error" in db_status:
        return "데이터베이스 연결에 문제가 있습니다. 시스템 관리자에게 문의하세요."

    if not db_status["has_documents"]:
        return (
            "문서가 업로드되지 않았습니다. 문서를 업로드한 후 RAG 시스템을 사용하세요."
        )

    if not db_status["has_embeddings"]:
        return "임베딩이 생성되지 않았습니다. 문서 처리가 완료될 때까지 기다리거나 관리자에게 문의하세요."

    if db_status["is_ready"]:
        return f"시스템이 정상 작동 중입니다. {db_status['document_count']}개 문서, {db_status['embedding_count']}개 임베딩이 준비되어 있습니다."

    return "데이터베이스 상태를 확인할 수 없습니다."


# 샘플 테스트 데이터
@router.get("/sample/")
async def get_sample_test():
    """샘플 테스트 데이터 반환."""
    return {
        "vector_search_query": {
            "url": "GET /rag/query/?question=FastAPI의 장점은 무엇인가요?&user_id=test_user",
            "description": "벡터 DB 검색 기반 RAG 질의 (폴백 메커니즘 포함)",
            "parameters": {
                "question": "FastAPI의 장점은 무엇인가요?",
                "user_id": "test_user",
                "max_documents": 5,
                "similarity_threshold": 0.7,
                "temperature": 0.1,
            },
        },
        "direct_answer": {
            "url": "POST /rag/answer/",
            "description": "컨텍스트가 주어진 상태에서 직접 답변 생성",
            "body": {
                "question": "FastAPI의 장점은 무엇인가요?",
                "context_documents": [
                    "FastAPI는 Python으로 작성된 현대적이고 빠른 웹 프레임워크입니다. 자동 API 문서 생성, 타입 힌트 지원, 높은 성능을 제공합니다.",
                    "FastAPI는 비동기 처리를 지원하여 높은 동시성을 제공합니다. 또한 Pydantic을 사용한 데이터 검증과 직렬화를 지원합니다.",
                ],
                "temperature": 0.1,
            },
        },
        "system_status": {
            "health_check": "GET /rag/health/",
            "database_status": "GET /rag/database/status/",
            "model_info": "GET /rag/model/info/",
        },
        "features": [
            "벡터 검색 실패 시 LLM 내재 지식으로 폴백",
            "DB 상태에 따른 적응적 응답",
            "동적 유사도 임계값 조정 (재검색)",
            "상세한 성능 메트릭스 (검색/생성 시간)",
        ],
    }
