"""임베딩 서비스 - 텍스트를 벡터로 변환."""

import logging
from typing import List

from sentence_transformers import SentenceTransformer

from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """한국어 텍스트 임베딩 서비스."""

    def __init__(self):
        self.model_name = settings.embedding_model
        self.dimension = settings.embedding_dimension
        self._model = None

    def _load_model(self):
        """임베딩 모델을 로드합니다."""
        if self._model is None:
            logger.info(f"임베딩 모델 로딩 중: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.info(f"임베딩 모델 로딩 완료: {self.dimension}차원")

    async def encode_text(self, text: str) -> List[float]:
        """단일 텍스트를 임베딩 벡터로 변환합니다."""
        if not text or not text.strip():
            raise ValueError("텍스트가 제공되지 않았습니다")

        try:
            self._load_model()

            logger.info(f"텍스트 임베딩 생성 중: {text[:50]}...")

            # 텍스트를 임베딩으로 변환
            embedding = self._model.encode(text.strip(), normalize_embeddings=True)

            # numpy array를 Python list로 변환
            embedding_list = embedding.tolist()

            logger.info(f"임베딩 생성 완료: {len(embedding_list)}차원")
            return embedding_list

        except Exception as e:
            logger.error(f"텍스트 임베딩 생성 중 오류: {str(e)}")
            raise Exception(f"임베딩 생성에 실패했습니다: {str(e)}")

    async def encode_texts(self, texts: List[str]) -> List[List[float]]:
        """여러 텍스트를 배치로 임베딩 벡터로 변환합니다."""
        if not texts:
            raise ValueError("텍스트 목록이 제공되지 않았습니다")

        try:
            self._load_model()

            # 빈 텍스트 필터링
            valid_texts = [text.strip() for text in texts if text and text.strip()]
            if not valid_texts:
                raise ValueError("유효한 텍스트가 없습니다")

            logger.info(f"배치 임베딩 생성 중: {len(valid_texts)}개 텍스트")

            # 배치로 임베딩 생성
            embeddings = self._model.encode(
                valid_texts, normalize_embeddings=True, batch_size=32
            )

            # numpy array를 Python list로 변환
            embeddings_list = [embedding.tolist() for embedding in embeddings]

            logger.info(f"배치 임베딩 생성 완료: {len(embeddings_list)}개")
            return embeddings_list

        except Exception as e:
            logger.error(f"배치 임베딩 생성 중 오류: {str(e)}")
            raise Exception(f"배치 임베딩 생성에 실패했습니다: {str(e)}")

    def get_model_info(self) -> dict:
        """임베딩 모델 정보를 반환합니다."""
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "loaded": self._model is not None,
        }
