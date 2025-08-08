"""Embedding domain models for business logic."""

import uuid
from datetime import datetime
from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EmbeddingModel(BaseModel):
    """임베딩 도메인 모델."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    chunk_id: uuid.UUID
    embedding: List[float] = Field(..., min_length=1, description="임베딩 벡터")
    created_at: datetime

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v: List[float]) -> List[float]:
        """임베딩 벡터 검증."""
        if not v:
            raise ValueError("임베딩 벡터는 비어있을 수 없습니다")

        # ko-sroberta-multitask 모델의 차원 확인 (768차원)
        expected_dim = 768
        if len(v) != expected_dim:
            raise ValueError(
                f"임베딩 차원이 잘못되었습니다. 기대값: {expected_dim}, 실제값: {len(v)}"
            )

        # 벡터의 각 요소가 유효한 float 값인지 확인
        for i, val in enumerate(v):
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"임베딩 벡터의 {i}번째 요소가 숫자가 아닙니다: {type(val)}"
                )
            if not (-1.0 <= val <= 1.0):
                # 일반적으로 정규화된 임베딩은 -1~1 범위, 하지만 모델에 따라 다를 수 있음
                # 너무 엄격하게 하지 않고 경고만 표시하도록 수정 가능
                pass

        return v

    def get_dimension(self) -> int:
        """임베딩 차원 수를 반환."""
        return len(self.embedding)

    def get_magnitude(self) -> float:
        """임베딩 벡터의 크기(magnitude)를 계산."""
        return sum(x * x for x in self.embedding) ** 0.5

    def is_normalized(self, tolerance: float = 1e-6) -> bool:
        """임베딩 벡터가 정규화되었는지 확인."""
        magnitude = self.get_magnitude()
        return abs(magnitude - 1.0) < tolerance

    def normalize(self) -> List[float]:
        """임베딩 벡터를 정규화하여 반환."""
        magnitude = self.get_magnitude()
        if magnitude == 0:
            raise ValueError("영벡터는 정규화할 수 없습니다")
        return [x / magnitude for x in self.embedding]

    def cosine_similarity(self, other: "EmbeddingModel") -> float:
        """다른 임베딩과의 코사인 유사도를 계산."""
        if len(self.embedding) != len(other.embedding):
            raise ValueError("임베딩 차원이 다릅니다")

        # 내적 계산
        dot_product = sum(a * b for a, b in zip(self.embedding, other.embedding))

        # 크기 계산
        magnitude_a = self.get_magnitude()
        magnitude_b = other.get_magnitude()

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)
