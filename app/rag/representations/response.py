"""Response models for RAG system."""

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class DocumentChunk(BaseModel):
    """문서 청크 응답 스키마."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    chunk_index: int = Field(..., description="청크 순서")
    content: str = Field(..., description="청크 내용")
    chunk_size: int = Field(..., description="청크 크기")
    created_at: datetime


class DocumentResponse(BaseModel):
    """문서 응답 스키마."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str = Field(..., description="문서 제목")
    content: str = Field(..., description="문서 내용")
    file_path: Optional[str] = Field(None, description="파일 경로")
    file_type: Optional[str] = Field(None, description="파일 타입")
    file_size: Optional[int] = Field(None, description="파일 크기")
    user_id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    chunks: Optional[List[DocumentChunk]] = Field(
        default=None, description="문서 청크들"
    )


class DocumentSummary(BaseModel):
    """문서 요약 응답 스키마."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str = Field(..., description="문서 제목")
    file_type: Optional[str] = Field(None, description="파일 타입")
    file_size: Optional[int] = Field(None, description="파일 크기")
    chunks_count: int = Field(..., description="청크 개수")
    created_at: datetime


class EmbeddingBase(BaseModel):
    """임베딩 기본 스키마."""

    embedding: List[float] = Field(..., description="임베딩 벡터")


class Embedding(EmbeddingBase):
    """임베딩 응답 스키마."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    chunk_id: uuid.UUID
    created_at: datetime


class RAGQueryResponse(BaseModel):
    """RAG 응답 결과 스키마."""

    answer: str = Field(..., description="생성된 답변")
    context_documents: List[str] = Field(default=[], description="참조한 문서 컨텍스트")
    confidence_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="신뢰도 점수"
    )
    sources: Optional[List[DocumentResponse]] = Field(
        default=None, description="참조 문서들"
    )


class RAGQuery(BaseModel):
    """RAG 질의 응답 기록 스키마."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    user_id: uuid.UUID
    question: str = Field(..., description="사용자 질문")
    answer: str = Field(..., description="생성된 답변")
    context_documents: Optional[List[str]] = Field(
        default=None, description="참조 문서 컨텍스트"
    )
    confidence_score: Optional[int] = Field(None, description="신뢰도 점수")
    feedback: Optional[str] = Field(None, description="사용자 피드백")
    created_at: datetime


class UploadResponse(BaseModel):
    """파일 업로드 응답 스키마."""

    document_id: uuid.UUID
    title: str = Field(..., description="업로드된 문서 제목")
    file_type: str = Field(..., description="파일 타입")
    file_size: int = Field(..., description="파일 크기")
    chunks_count: int = Field(..., description="생성된 청크 개수")
    message: str = Field(..., description="업로드 결과 메시지")


class RAGResponse(BaseModel):
    """RAG 응답 스키마."""

    question: str
    answer: str
    context_count: int
    model_info: dict


class HealthResponse(BaseModel):
    """상태 확인 응답 스키마."""

    status: str
    components: dict


class DocumentSource(BaseModel):
    """검색된 문서 출처 정보."""

    document_id: str
    chunk_index: int
    content: str
    similarity_score: float


class RAGQueryResponse(BaseModel):
    """Vector DB 기반 RAG 질의 응답 스키마."""

    question: str = Field(..., description="사용자 질문")
    answer: str = Field(..., description="생성된 답변")
    sources: List[DocumentSource] = Field(default=[], description="참조 문서 출처들")
    confidence_score: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="답변 신뢰도"
    )
    search_time_ms: Optional[int] = Field(None, description="검색 소요 시간 (ms)")
    generation_time_ms: Optional[int] = Field(
        None, description="답변 생성 소요 시간 (ms)"
    )

    # 폴백 모드 관련 필드들
    fallback_mode: Optional[bool] = Field(
        False, description="폴백 모드 여부 (벡터 검색 실패 시 일반 지식 사용)"
    )
    retry_attempted: Optional[bool] = Field(
        False, description="낮은 임계값으로 재검색 시도 여부"
    )
    retry_threshold: Optional[float] = Field(
        None, description="재검색 시 사용된 임계값"
    )
    db_status: Optional[dict] = Field(None, description="데이터베이스 상태 정보")
    error: Optional[bool] = Field(False, description="오류 발생 여부")
