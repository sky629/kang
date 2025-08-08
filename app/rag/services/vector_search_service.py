"""Vector 검색 서비스 - pgvector를 활용한 유사도 검색."""

import logging
from typing import List, Optional

from sqlalchemy import text

from app.common.storage.postgres import postgres_storage
from app.rag.models.postgres_models import DocumentChunk, Embedding
from config.settings import settings

logger = logging.getLogger(__name__)


class DocumentChunkResult:
    """검색 결과 문서 청크."""

    def __init__(self, chunk: DocumentChunk, similarity_score: float):
        self.chunk = chunk
        self.similarity_score = similarity_score
        self.content = chunk.content
        self.document_id = chunk.document_id
        self.chunk_index = chunk.chunk_index


class VectorSearchService:
    """pgvector를 활용한 문서 유사도 검색 서비스."""

    def __init__(self):
        self.similarity_threshold = settings.similarity_threshold
        self.max_retrieved_docs = settings.max_retrieved_docs

    async def search_similar_documents(
        self,
        query_embedding: List[float],
        similarity_threshold: Optional[float] = None,
        max_docs: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> List[DocumentChunkResult]:
        """쿼리 임베딩과 유사한 문서 청크들을 검색합니다."""

        if not query_embedding:
            raise ValueError("쿼리 임베딩이 제공되지 않았습니다")

        # 설정값 사용 또는 기본값 적용
        threshold = similarity_threshold or self.similarity_threshold
        limit = max_docs or self.max_retrieved_docs

        try:
            async with postgres_storage.get_domain_read_session("rag") as session:
                logger.info(f"벡터 검색 시작 - threshold: {threshold}, limit: {limit}")

                # pgvector 코사인 유사도 검색 쿼리
                query = text(
                    """
                    SELECT 
                        dc.id,
                        dc.document_id,
                        dc.chunk_index,
                        dc.content,
                        dc.chunk_size,
                        dc.created_at,
                        (1 - (e.embedding <=> :query_embedding)) AS similarity_score
                    FROM document_chunks dc
                    JOIN embeddings e ON dc.id = e.chunk_id
                    JOIN documents d ON dc.document_id = d.id
                    WHERE (1 - (e.embedding <=> :query_embedding)) >= :threshold
                    ORDER BY similarity_score DESC
                    LIMIT :limit
                """
                )

                # 쿼리 실행
                result = await session.execute(
                    query,
                    {
                        "query_embedding": query_embedding,
                        "threshold": threshold,
                        "limit": limit,
                    },
                )

                rows = result.fetchall()

                if not rows:
                    logger.info("검색 결과 없음")
                    return []

                # 결과 변환
                search_results = []
                for row in rows:
                    # DocumentChunk 객체 생성
                    chunk = DocumentChunk(
                        id=row.id,
                        document_id=row.document_id,
                        chunk_index=row.chunk_index,
                        content=row.content,
                        chunk_size=row.chunk_size,
                        created_at=row.created_at,
                    )

                    # 검색 결과 객체 생성
                    search_result = DocumentChunkResult(
                        chunk=chunk, similarity_score=float(row.similarity_score)
                    )
                    search_results.append(search_result)

                logger.info(f"벡터 검색 완료: {len(search_results)}개 문서 발견")
                return search_results

        except Exception as e:
            logger.error(f"벡터 검색 중 오류: {str(e)}")
            raise Exception(f"문서 검색에 실패했습니다: {str(e)}")

    async def get_document_chunks_by_ids(
        self, chunk_ids: List[str]
    ) -> List[DocumentChunk]:
        """청크 ID 목록으로 문서 청크들을 가져옵니다."""

        if not chunk_ids:
            return []

        try:
            async with postgres_storage.get_domain_read_session("rag") as session:
                result = await session.execute(
                    text(
                        """
                        SELECT dc.*
                        FROM document_chunks dc
                        WHERE dc.id = ANY(:chunk_ids)
                        ORDER BY dc.chunk_index
                    """
                    ),
                    {"chunk_ids": chunk_ids},
                )

                rows = result.fetchall()

                chunks = []
                for row in rows:
                    chunk = DocumentChunk(
                        id=row.id,
                        document_id=row.document_id,
                        chunk_index=row.chunk_index,
                        content=row.content,
                        chunk_size=row.chunk_size,
                        created_at=row.created_at,
                    )
                    chunks.append(chunk)

                return chunks

        except Exception as e:
            logger.error(f"청크 조회 중 오류: {str(e)}")
            raise Exception(f"문서 청크 조회에 실패했습니다: {str(e)}")

    def get_search_info(self) -> dict:
        """검색 설정 정보를 반환합니다."""
        return {
            "similarity_threshold": self.similarity_threshold,
            "max_retrieved_docs": self.max_retrieved_docs,
            "embedding_dimension": settings.embedding_dimension,
        }
