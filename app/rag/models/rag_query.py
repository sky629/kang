"""RAG Query domain models for business logic."""

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RAGQueryModel(BaseModel):
    """RAG 질의 도메인 모델."""
    model_config = ConfigDict(from_attributes=True)
    
    id: uuid.UUID
    user_id: uuid.UUID
    question: str = Field(..., min_length=1, max_length=1000, description="사용자 질문")
    answer: str = Field(..., min_length=1, description="생성된 답변")
    context_documents: Optional[List[str]] = Field(default=None, description="참조된 문서 ID들")
    confidence_score: Optional[int] = Field(None, ge=1, le=10, description="신뢰도 점수 (1-10)")
    feedback: Optional[str] = Field(None, max_length=500, description="사용자 피드백")
    created_at: datetime
    
    @field_validator('question')
    @classmethod
    def validate_question(cls, v: str) -> str:
        """질문 검증."""
        if not v.strip():
            raise ValueError("질문은 비어있을 수 없습니다")
        
        # 질문 길이 검증
        if len(v.strip()) < 2:
            raise ValueError("질문은 최소 2자 이상이어야 합니다")
        
        return v.strip()
    
    @field_validator('answer')
    @classmethod
    def validate_answer(cls, v: str) -> str:
        """답변 검증."""
        if not v.strip():
            raise ValueError("답변은 비어있을 수 없습니다")
        
        return v.strip()
    
    @field_validator('context_documents')
    @classmethod
    def validate_context_documents(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """참조 문서 ID 목록 검증."""
        if v is not None:
            # 빈 리스트는 허용
            if not isinstance(v, list):
                raise ValueError("context_documents는 문자열 리스트여야 합니다")
            
            # 각 문서 ID가 유효한 UUID 형식인지 확인
            for doc_id in v:
                if not isinstance(doc_id, str):
                    raise ValueError("문서 ID는 문자열이어야 합니다")
                try:
                    uuid.UUID(doc_id)  # UUID 형식 검증
                except ValueError:
                    raise ValueError(f"유효하지 않은 문서 ID 형식입니다: {doc_id}")
        
        return v
    
    @field_validator('feedback')
    @classmethod
    def validate_feedback(cls, v: Optional[str]) -> Optional[str]:
        """피드백 검증."""
        if v is not None:
            return v.strip() if v.strip() else None
        return v
    
    def get_context_documents_count(self) -> int:
        """참조된 문서 개수를 반환."""
        return len(self.context_documents) if self.context_documents else 0
    
    def has_high_confidence(self, threshold: int = 7) -> bool:
        """높은 신뢰도를 가지는지 확인."""
        return self.confidence_score is not None and self.confidence_score >= threshold
    
    def has_feedback(self) -> bool:
        """사용자 피드백이 있는지 확인."""
        return self.feedback is not None and len(self.feedback.strip()) > 0
    
    def is_satisfied(self, min_confidence: int = 6) -> bool:
        """만족스러운 답변인지 판단 (신뢰도 기준)."""
        if self.confidence_score is None:
            return False
        return self.confidence_score >= min_confidence
    
    def get_quality_score(self) -> float:
        """답변 품질 점수를 계산 (0.0 ~ 1.0)."""
        score = 0.0
        
        # 신뢰도 점수 반영 (70%)
        if self.confidence_score is not None:
            score += (self.confidence_score / 10.0) * 0.7
        
        # 참조 문서 수 반영 (20%)
        context_count = self.get_context_documents_count()
        if context_count > 0:
            # 1-5개 문서는 좋음, 그 이상은 너무 많음
            context_score = min(context_count / 5.0, 1.0)
            score += context_score * 0.2
        
        # 답변 길이 반영 (10%)
        answer_length = len(self.answer)
        if answer_length >= 50:  # 최소 50자 이상
            length_score = min(answer_length / 500.0, 1.0)  # 500자를 최적으로 봄
            score += length_score * 0.1
        
        return min(score, 1.0)  # 최대 1.0으로 제한