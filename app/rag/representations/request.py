"""Request models for RAG system."""

from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentBase(BaseModel):
    """문서 기본 스키마."""
    title: str = Field(..., min_length=1, max_length=255, description="문서 제목")
    content: str = Field(..., min_length=1, description="문서 내용")
    file_type: Optional[str] = Field(None, max_length=50, description="파일 타입")
    file_size: Optional[int] = Field(None, gt=0, description="파일 크기 (bytes)")


class DocumentCreateRequest(DocumentBase):
    """문서 생성 요청 스키마."""
    pass


class DocumentUpdateRequest(BaseModel):
    """문서 업데이트 요청 스키마."""
    title: Optional[str] = Field(None, min_length=1, max_length=255, description="문서 제목")
    content: Optional[str] = Field(None, min_length=1, description="문서 내용")


class RAGQueryBase(BaseModel):
    """RAG 질의 기본 스키마."""
    question: str = Field(..., min_length=1, max_length=1000, description="질문 내용")


class RAGQueryRequest(RAGQueryBase):
    """RAG 질의 요청 스키마."""
    pass


class RAGFeedbackRequest(BaseModel):
    """RAG 질의 피드백 요청 스키마."""
    confidence_score: Optional[int] = Field(
        None, ge=1, le=10, description="신뢰도 점수 (1-10)"
    )
    feedback: Optional[str] = Field(
        None, max_length=500, description="사용자 피드백"
    )


class RAGRequest(BaseModel):
    """RAG 요청 스키마."""
    question: str = Field(..., min_length=1, description="질문")
    context_documents: List[str] = Field(..., min_items=1, description="참조 문서들")
    temperature: float = Field(0.1, ge=0.0, le=1.0, description="응답 창의성")