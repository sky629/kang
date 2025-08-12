"""GPT-OSS service for text generation using Ollama with direct HTTP calls."""

import logging
from typing import Dict, List, Optional

import httpx

from config.settings import settings

logger = logging.getLogger(__name__)


class GPTOSSService:
    """GPT-OSS 로컬 모델을 통한 텍스트 생성 서비스 - 직접 HTTP 호출."""

    def __init__(self):
        self.base_url = settings.gpt_oss_base_url
        self.model = settings.gpt_oss_model
        self.max_tokens = settings.gpt_oss_max_tokens
        self.temperature = settings.gpt_oss_temperature
        self.reasoning_level = settings.gpt_oss_reasoning_level
        self.timeout = settings.gpt_oss_timeout

    async def _make_request(
        self, prompt: str, system_prompt: Optional[str] = None, **kwargs
    ) -> str:
        """Ollama API에 직접 HTTP 요청을 보내고 응답을 받습니다."""

        try:
            # 최종 프롬프트 구성
            if system_prompt:
                final_prompt = f"System: {system_prompt}\n\nHuman: {prompt}\n\nAssistant:"
            else:
                final_prompt = prompt

            # Ollama API 요청 데이터
            request_data = {
                "model": self.model,
                "prompt": final_prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", self.temperature),
                    "num_predict": kwargs.get("max_tokens", self.max_tokens),
                }
            }

            logger.info(f"Ollama API 요청 전송: {self.base_url}/api/generate")

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=request_data
                )
                
                if response.status_code != 200:
                    raise Exception(f"HTTP {response.status_code}: {response.text}")

                result = response.json()
                
                if not result.get("done", False):
                    raise Exception("응답이 완료되지 않았습니다")

                generated_text = result.get("response", "").strip()
                
                if not generated_text:
                    raise Exception("빈 응답을 받았습니다")

                logger.info("Ollama API 응답 수신 완료")
                return generated_text

        except Exception as e:
            logger.error(f"Ollama API 요청 처리 중 오류: {str(e)}")
            raise Exception(f"LLM 요청에 실패했습니다: {str(e)}")

    def _build_rag_system_prompt(self) -> str:
        """RAG 시스템용 시스템 프롬프트를 생성합니다."""
        return """당신은 한국어 문서 기반 질의응답 전문가입니다.
주어진 참조 문서들을 바탕으로 정확하고 도움이 되는 답변을 제공해주세요.

지침:
1. 참조 문서의 내용만을 기반으로 답변하세요
2. 참조 문서에 없는 정보는 추측하지 마세요
3. 답변은 자연스러운 한국어로 작성하세요
4. 가능한 한 구체적이고 상세한 답변을 제공하세요
5. 불확실한 경우 "참조 문서에서 명확한 정보를 찾을 수 없습니다"라고 말하세요"""

    def _build_rag_prompt(self, question: str, context_documents: List[str]) -> str:
        """RAG용 프롬프트를 생성합니다."""
        
        # 컨텍스트 문서들을 포맷팅
        context_text = "\n\n".join([
            f"[참조 문서 {i+1}]\n{doc}" 
            for i, doc in enumerate(context_documents)
        ])
        
        return f"""참조 문서들:
{context_text}

질문: {question}

위의 참조 문서들을 바탕으로 질문에 대해 정확하고 도움이 되는 답변을 제공해주세요."""

    async def generate_answer(
        self,
        question: str,
        context_documents: List[str],
        user_id: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """컨텍스트 문서를 기반으로 RAG 답변을 생성합니다."""

        if not question or not question.strip():
            raise ValueError("질문이 제공되지 않았습니다")

        if not context_documents:
            raise ValueError("참조 문서가 제공되지 않았습니다")

        if not user_id or not user_id.strip():
            raise ValueError("사용자 ID가 제공되지 않았습니다")

        try:
            logger.info(f"RAG 답변 생성 시작 - 사용자: {user_id}")
            logger.info(f"질문: {question[:100]}...")
            logger.info(f"참조 문서 수: {len(context_documents)}")

            # RAG 프롬프트 생성
            system_prompt = self._build_rag_system_prompt()
            rag_prompt = self._build_rag_prompt(question, context_documents)

            # LLM 호출
            answer = await self._make_request(
                prompt=rag_prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            logger.info("RAG 답변 생성 완료")
            return answer

        except Exception as e:
            logger.error(f"RAG 답변 생성 중 오류: {str(e)}")
            raise Exception(f"답변 생성에 실패했습니다: {str(e)}")

    async def generate_rag_answer(
        self,
        question: str,
        context_documents: List[str],
        user_id: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """RAG 답변 생성 (호환성을 위한 별칭 메서드)."""
        return await self.generate_answer(
            question=question,
            context_documents=context_documents,
            user_id=user_id or "unknown",
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    async def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """일반적인 텍스트 생성."""

        if not prompt or not prompt.strip():
            raise ValueError("프롬프트가 제공되지 않았습니다")

        try:
            logger.info("일반 텍스트 생성 시작")
            logger.info(f"프롬프트: {prompt[:100]}...")

            response = await self._make_request(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )

            logger.info("일반 텍스트 생성 완료")
            return response

        except Exception as e:
            logger.error(f"일반 텍스트 생성 중 오류: {str(e)}")
            raise Exception(f"텍스트 생성에 실패했습니다: {str(e)}")

    async def check_health(self) -> bool:
        """GPT-OSS 서비스 상태를 확인합니다."""
        try:
            logger.info("GPT-OSS 서비스 상태 확인 시작")

            async with httpx.AsyncClient(timeout=10) as client:
                # Ollama 서버 상태 확인
                response = await client.get(f"{self.base_url}/api/tags")
                
                if response.status_code != 200:
                    logger.error(f"Ollama 서버 연결 실패: HTTP {response.status_code}")
                    return False

                # 사용 가능한 모델 목록 확인
                models_data = response.json()
                available_models = [model["name"] for model in models_data.get("models", [])]

                if self.model not in available_models:
                    logger.error(f"모델 {self.model}을 찾을 수 없습니다. 사용 가능한 모델: {available_models}")
                    return False

                logger.info("GPT-OSS 서비스 상태 확인 완료 - 정상")
                return True

        except Exception as e:
            logger.error(f"GPT-OSS 서비스 상태 확인 중 오류: {str(e)}")
            return False

    async def pull_model(self) -> bool:
        """모델을 다운로드합니다."""
        try:
            logger.info(f"모델 다운로드 시작: {self.model}")

            async with httpx.AsyncClient(timeout=300) as client:  # 5분 타임아웃
                request_data = {"name": self.model}
                
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json=request_data
                )
                
                if response.status_code != 200:
                    logger.error(f"모델 다운로드 실패: HTTP {response.status_code}")
                    return False

                logger.info("모델 다운로드 완료")
                return True

        except Exception as e:
            logger.error(f"모델 다운로드 중 오류: {str(e)}")
            return False

    def get_model_info(self) -> Dict[str, any]:
        """모델 정보를 반환합니다."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "reasoning_level": self.reasoning_level,
            "timeout": self.timeout,
            "service_type": "Ollama HTTP Client",
        }