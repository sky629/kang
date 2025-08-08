"""Document domain models for business logic."""

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DocumentChunkModel(BaseModel):
    """문서 청크 도메인 모델."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int = Field(..., ge=0, description="청크 순서")
    content: str = Field(..., min_length=1, description="청크 내용")
    chunk_size: int = Field(..., gt=0, description="청크 크기")
    created_at: datetime

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """청크 내용 검증."""
        if not v.strip():
            raise ValueError("청크 내용은 비어있을 수 없습니다")
        return v.strip()


class DocumentModel(BaseModel):
    """문서 도메인 모델."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str = Field(..., min_length=1, max_length=255, description="문서 제목")
    content: str = Field(..., min_length=1, description="문서 내용")
    file_path: Optional[str] = Field(None, max_length=500, description="파일 경로")
    file_type: Optional[str] = Field(None, max_length=50, description="파일 타입")
    file_size: Optional[int] = Field(None, gt=0, description="파일 크기 (bytes)")
    user_id: uuid.UUID
    created_at: datetime
    updated_at: datetime
    chunks: Optional[List[DocumentChunkModel]] = Field(
        default=None, description="문서 청크들"
    )

    @field_validator("title")
    @classmethod
    def validate_title(cls, v: str) -> str:
        """문서 제목 검증."""
        if not v.strip():
            raise ValueError("문서 제목은 비어있을 수 없습니다")
        return v.strip()

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        """문서 내용 검증."""
        if not v.strip():
            raise ValueError("문서 내용은 비어있을 수 없습니다")
        return v.strip()

    @field_validator("file_type")
    @classmethod
    def validate_file_type(cls, v: Optional[str]) -> Optional[str]:
        """파일 타입 검증."""
        if v is not None:
            allowed_types = {"pdf", "docx", "txt", "md"}
            if v.lower() not in allowed_types:
                raise ValueError(
                    f"지원되지 않는 파일 타입입니다: {v}. 지원 타입: {allowed_types}"
                )
            return v.lower()
        return v

    def get_total_chunks_size(self) -> int:
        """모든 청크의 총 크기를 반환."""
        if not self.chunks:
            return 0
        return sum(chunk.chunk_size for chunk in self.chunks)

    def get_chunks_count(self) -> int:
        """청크 개수를 반환."""
        return len(self.chunks) if self.chunks else 0

    def is_chunked(self) -> bool:
        """문서가 청크로 분할되었는지 확인."""
        return self.chunks is not None and len(self.chunks) > 0
