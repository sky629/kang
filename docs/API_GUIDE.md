# RAG API 사용 가이드

**마지막 업데이트**: 2025-01-08  
**API 버전**: v1.0 (Pydantic 모델 기반)

## 🚀 빠른 시작

### 기본 RAG 질의
```bash
curl -X POST "http://localhost:8000/rag/query/" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "FastAPI의 주요 장점은 무엇인가요?",
    "user_id": "user123",
    "max_documents": 5,
    "similarity_threshold": 0.7,
    "temperature": 0.1
  }'
```

### 응답 예시
```json
{
  "question": "FastAPI의 주요 장점은 무엇인가요?",
  "answer": "FastAPI는 현대적이고 고성능인 웹 프레임워크입니다...",
  "confidence_score": 0.85,
  "search_time_ms": 150.2,
  "generation_time_ms": 1200.5,
  "fallback_mode": false,
  "retry_attempted": false,
  "retrieved_documents": 3,
  "db_status": {
    "document_count": 25,
    "embedding_count": 150,
    "is_ready": true
  }
}
```

## 📋 API 엔드포인트

### 1. POST /rag/query/ - 메인 RAG 질의응답

**요청 모델**: `RAGQueryParametersRequest`

| 필드 | 타입 | 기본값 | 제약조건 | 설명 |
|------|------|--------|----------|------|
| `question` | string | - | 1-1000자 | **필수** 사용자 질문 |
| `user_id` | string | "default" | - | 사용자 식별자 |
| `max_documents` | integer | 5 | 1-10 | 최대 검색 문서 수 |
| `similarity_threshold` | float | 0.7 | 0.0-1.0 | 유사도 임계값 |
| `temperature` | float | 0.1 | 0.0-1.0 | LLM 응답 창의성 |

**응답 모델**: `RAGQueryResponse`

### 2. POST /rag/answer/ - 직접 컨텍스트 기반 답변

직접 제공된 문서로 답변을 생성합니다.

```bash
curl -X POST "http://localhost:8000/rag/answer/" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "FastAPI의 장점은?",
    "context_documents": [
      "FastAPI는 Python으로 작성된 현대적이고 빠른 웹 프레임워크입니다.",
      "자동 API 문서 생성과 타입 힌트를 지원합니다."
    ],
    "temperature": 0.1
  }'
```

### 3. GET /rag/health/ - 시스템 상태 확인

```bash
curl "http://localhost:8000/rag/health/"
```

시스템 전반적인 상태를 확인합니다:
- DB 연결 상태
- 임베딩 모델 로딩 상태
- LLM 서비스 상태

### 4. GET /rag/database/status/ - 데이터베이스 상태

```bash
curl "http://localhost:8000/rag/database/status/"
```

벡터 데이터베이스의 현재 상태와 권장사항을 제공합니다.

### 5. GET /rag/sample/ - 사용법 예시

```bash
curl "http://localhost:8000/rag/sample/"
```

API 사용 예시와 기능 설명을 제공합니다.

## 🧠 지능적 폴백 메커니즘

### 동적 임계값 조정
RAG 시스템은 검색 결과가 없을 때 자동으로 임계값을 낮춰 재검색을 시도합니다:

1. **1차 시도**: `similarity_threshold = 0.7`
2. **2차 시도**: `similarity_threshold = 0.5`  
3. **3차 시도**: `similarity_threshold = 0.3`
4. **폴백**: LLM 일반 지식 활용

### 신뢰도 점수 해석
- **0.8-1.0**: 매우 높은 신뢰도 (정확한 문서 매칭)
- **0.6-0.8**: 높은 신뢰도 (관련성 있는 내용)
- **0.4-0.6**: 보통 신뢰도 (부분적 관련성)
- **0.2-0.4**: 낮은 신뢰도 (폴백 모드)
- **0.0-0.2**: 매우 낮음 (시스템 오류 가능성)

## 🔧 고급 사용법

### Python 클라이언트 예시

```python
import httpx
import asyncio

async def query_rag(question: str, user_id: str = "default"):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/rag/query/",
            json={
                "question": question,
                "user_id": user_id,
                "max_documents": 5,
                "similarity_threshold": 0.7,
                "temperature": 0.1
            }
        )
        return response.json()

# 사용 예시
result = asyncio.run(query_rag("FastAPI의 장점은 무엇인가요?"))
print(f"답변: {result['answer']}")
print(f"신뢰도: {result['confidence_score']}")
print(f"검색 시간: {result['search_time_ms']}ms")
```

### JavaScript/Node.js 예시

```javascript
const axios = require('axios');

async function queryRAG(question, userId = 'default') {
  try {
    const response = await axios.post('http://localhost:8000/rag/query/', {
      question: question,
      user_id: userId,
      max_documents: 5,
      similarity_threshold: 0.7,
      temperature: 0.1
    });
    
    return response.data;
  } catch (error) {
    console.error('RAG API 호출 실패:', error.response.data);
    throw error;
  }
}

// 사용 예시
queryRAG('FastAPI의 장점은 무엇인가요?')
  .then(result => {
    console.log('답변:', result.answer);
    console.log('신뢰도:', result.confidence_score);
    console.log('폴백 모드:', result.fallback_mode);
  });
```

## ⚠️ 오류 처리

### 일반적인 오류 코드

| 상태 코드 | 설명 | 해결 방법 |
|-----------|------|-----------|
| 400 | 잘못된 요청 (유효성 검증 실패) | 요청 파라미터 확인 |
| 500 | 서버 내부 오류 | 시스템 상태 확인 |
| 503 | 서비스 일시 불가 | 잠시 후 재시도 |

### 오류 응답 예시

```json
{
  "detail": "Field validation error: question field is required"
}
```

## 📊 성능 최적화 팁

### 1. 임계값 설정
- **높은 정확도**: `similarity_threshold: 0.8`
- **균형**: `similarity_threshold: 0.7` (기본값)  
- **넓은 검색**: `similarity_threshold: 0.5`

### 2. 문서 수 조절
- **빠른 응답**: `max_documents: 3`
- **균형**: `max_documents: 5` (기본값)
- **상세 답변**: `max_documents: 10`

### 3. 온도 설정
- **정확한 답변**: `temperature: 0.1` (기본값)
- **창의적 답변**: `temperature: 0.7`
- **매우 창의적**: `temperature: 1.0`

## 🔍 모니터링 및 디버깅

### 성능 메트릭 활용
```python
# 성능 모니터링 예시
result = await query_rag("질문")

if result['search_time_ms'] > 500:
    print("⚠️ 검색 시간이 느립니다. DB 인덱스를 확인하세요.")
    
if result['generation_time_ms'] > 3000:
    print("⚠️ LLM 응답이 느립니다. Ollama 상태를 확인하세요.")
    
if result['fallback_mode']:
    print("ℹ️ 폴백 모드로 동작 중입니다. 문서를 업로드하세요.")
```

### 시스템 상태 모니터링
```bash
# 정기적으로 실행하여 시스템 상태 확인
curl "http://localhost:8000/rag/health/" | jq '.status'
curl "http://localhost:8000/rag/database/status/" | jq '.status.is_ready'
```

---

**📚 추가 자료**
- [RAG 구현 상태](/docs/RAG_IMPLEMENTATION_STATUS.md)
- [RAG 베스트 프랙티스](/docs/RAG_BEST_PRACTICES.md)  
- [개발자 가이드](/CLAUDE.md)