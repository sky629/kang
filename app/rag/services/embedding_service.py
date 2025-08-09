"""임베딩 서비스 - 텍스트를 벡터로 변환."""

import logging
import os
from typing import List

# 임시 mock 클래스들 (M1 맥에서는 실제 클래스 사용)
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    # Mock HuggingFaceEmbeddings for development
    class HuggingFaceEmbeddings:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
        
        def embed_query(self, text):
            return [0.1] * 768  # Mock 768-dim embedding
        
        def embed_documents(self, texts):
            return [[0.1] * 768] * len(texts)  # Mock embeddings

from config.settings import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """한국어 텍스트 임베딩 서비스 - LangChain 기반."""

    def __init__(self):
        self.model_name = settings.embedding_model
        self.dimension = settings.embedding_dimension
        self._model = None
        
        # LangSmith 추적 설정
        if settings.langchain_tracing_v2 and settings.langchain_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2)
            os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
            os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
            os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

    def _load_model(self):
        """임베딩 모델을 로드합니다."""
        if self._model is None:
            logger.info(f"임베딩 모델 로딩 중: {self.model_name}")
            
            # HuggingFaceEmbeddings 사용
            model_kwargs = {
                'device': 'cpu',  # GPU 사용 시 'cuda'로 변경
                'normalize_embeddings': True
            }
            
            self._model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs=model_kwargs,
                encode_kwargs={'normalize_embeddings': True}
            )
            
            logger.info(f"임베딩 모델 로딩 완료: {self.dimension}차원")

    async def encode_text(self, text: str) -> List[float]:
        """단일 텍스트를 임베딩 벡터로 변환합니다."""
        if not text or not text.strip():
            raise ValueError("텍스트가 제공되지 않았습니다")

        try:
            self._load_model()

            logger.info(f"텍스트 임베딩 생성 중: {text[:50]}...")

            # LangChain HuggingFaceEmbeddings 사용
            embedding_list = self._model.embed_query(text.strip())

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

            # LangChain HuggingFaceEmbeddings로 배치 임베딩 생성
            embeddings_list = self._model.embed_documents(valid_texts)

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
