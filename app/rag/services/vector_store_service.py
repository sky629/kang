"""LangChain PGVector 기반 벡터 스토어 서비스."""

import logging
import os
from typing import List, Optional, Dict, Any

from langchain_postgres import PGVector

from app.rag.services.embedding_service import EmbeddingService
from config.settings import settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """LangChain PGVector를 활용한 벡터 스토어 서비스."""

    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.collection_name = "document_embeddings"
        self._vector_store = None
        
        # LangSmith 추적 설정
        if settings.langchain_tracing_v2 and settings.langchain_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = str(settings.langchain_tracing_v2)
            os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
            os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
            os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project

    def _get_vector_store(self) -> PGVector:
        """PGVector 인스턴스를 생성하고 반환합니다."""
        if self._vector_store is None:
            logger.info("PGVector 인스턴스 생성 중...")
            
            # 임베딩 서비스 로드
            self.embedding_service._load_model()
            
            # PGVector 설정
            connection_string = settings.postgres_url
            
            self._vector_store = PGVector(
                embeddings=self.embedding_service._model,
                collection_name=self.collection_name,
                connection=connection_string,
                use_jsonb=True,  # 메타데이터를 JSONB로 저장
            )
            
            logger.info("PGVector 인스턴스 생성 완료")
        
        return self._vector_store

    async def search_similar_documents(
        self,
        query: str,
        similarity_threshold: Optional[float] = None,
        max_docs: Optional[int] = None,
        user_id: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """쿼리와 유사한 문서들을 검색합니다."""
        
        if not query or not query.strip():
            raise ValueError("검색 쿼리가 제공되지 않았습니다")

        threshold = similarity_threshold or settings.similarity_threshold
        limit = max_docs or settings.max_retrieved_docs

        try:
            vector_store = self._get_vector_store()
            
            logger.info(f"벡터 검색 시작 - query: {query[:50]}..., threshold: {threshold}, limit: {limit}")

            # 메타데이터 필터 적용 (필요한 경우)
            search_kwargs = {"k": limit}
            if filter_metadata:
                search_kwargs["filter"] = filter_metadata

            # 유사도 검색 실행 (점수와 함께)
            docs_with_scores = vector_store.similarity_search_with_score(
                query=query,
                **search_kwargs
            )

            # 임계값 필터링
            filtered_results = [
                (doc, score) for doc, score in docs_with_scores 
                if (1.0 - score) >= threshold  # PGVector는 distance를 반환하므로 similarity로 변환
            ]

            if not filtered_results:
                logger.info("임계값을 만족하는 검색 결과 없음")
                return []

            # 결과 변환
            search_results = []
            for doc, score in filtered_results:
                similarity_score = 1.0 - score  # distance를 similarity로 변환
                
                result = {
                    "content": doc.page_content,
                    "similarity_score": similarity_score,
                    "metadata": doc.metadata or {},
                    "document_id": doc.metadata.get("document_id", ""),
                    "chunk_index": doc.metadata.get("chunk_index", 0),
                }
                search_results.append(result)

            logger.info(f"벡터 검색 완료: {len(search_results)}개 문서 발견")
            return search_results

        except Exception as e:
            logger.error(f"벡터 검색 중 오류: {str(e)}")
            raise Exception(f"문서 검색에 실패했습니다: {str(e)}")

    async def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """문서들을 벡터 스토어에 추가합니다."""
        
        if not texts:
            raise ValueError("추가할 텍스트가 제공되지 않았습니다")

        try:
            vector_store = self._get_vector_store()
            
            logger.info(f"문서 추가 중: {len(texts)}개 텍스트")

            # 문서 추가
            document_ids = vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids,
            )

            logger.info(f"문서 추가 완료: {len(document_ids)}개 문서")
            return document_ids

        except Exception as e:
            logger.error(f"문서 추가 중 오류: {str(e)}")
            raise Exception(f"문서 추가에 실패했습니다: {str(e)}")

    async def delete_documents(self, ids: List[str]) -> bool:
        """문서들을 벡터 스토어에서 삭제합니다."""
        
        if not ids:
            return True

        try:
            vector_store = self._get_vector_store()
            
            logger.info(f"문서 삭제 중: {len(ids)}개 문서")

            # 문서 삭제
            vector_store.delete(ids=ids)

            logger.info(f"문서 삭제 완료")
            return True

        except Exception as e:
            logger.error(f"문서 삭제 중 오류: {str(e)}")
            raise Exception(f"문서 삭제에 실패했습니다: {str(e)}")

    async def check_database_status(self) -> Dict[str, Any]:
        """벡터 데이터베이스 상태를 확인합니다."""
        try:
            vector_store = self._get_vector_store()
            
            # PGVector의 내부 연결을 통해 상태 확인
            # 실제 구현에서는 vector_store의 메서드를 사용하거나
            # 직접 DB 쿼리를 실행해야 할 수 있습니다
            
            # 임시로 기본 상태 정보 반환
            return {
                "vector_store_ready": True,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_service.model_name,
                "embedding_dimension": self.embedding_service.dimension,
                "status": "connected",
            }

        except Exception as e:
            logger.error(f"벡터 데이터베이스 상태 확인 중 오류: {str(e)}")
            return {
                "vector_store_ready": False,
                "collection_name": self.collection_name,
                "status": "error",
                "error": str(e),
            }

    def get_search_info(self) -> Dict[str, Any]:
        """벡터 검색 설정 정보를 반환합니다."""
        return {
            "similarity_threshold": settings.similarity_threshold,
            "max_retrieved_docs": settings.max_retrieved_docs,
            "embedding_dimension": settings.embedding_dimension,
            "embedding_model": self.embedding_service.model_name,
            "collection_name": self.collection_name,
        }