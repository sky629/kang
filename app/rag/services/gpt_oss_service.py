"""GPT-OSS service for text generation using Ollama."""

import json
import logging
from typing import Dict, List, Optional

import httpx
from config.settings import settings

logger = logging.getLogger(__name__)


class GPTOSSService:
    """GPT-OSS 로컬 모델을 통한 텍스트 생성 서비스."""
    
    def __init__(self):
        self.base_url = settings.gpt_oss_base_url
        self.model = settings.gpt_oss_model
        self.max_tokens = settings.gpt_oss_max_tokens
        self.temperature = settings.gpt_oss_temperature
        self.reasoning_level = settings.gpt_oss_reasoning_level
        self.timeout = settings.gpt_oss_timeout
        
    async def _make_request(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """Ollama API에 요청을 보내고 응답을 받습니다."""
        
        # Harmony 형식 프롬프트 구성
        full_prompt = self._build_harmony_prompt(prompt, system_prompt)
        
        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
                "stop": kwargs.get("stop_sequences", []),
            }
        }
        
        # 추론 레벨 설정 (GPT-OSS 특화 옵션)
        if self.reasoning_level:
            payload["options"]["reasoning_level"] = self.reasoning_level
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                logger.info(f"GPT-OSS 요청 전송: {self.base_url}/api/generate")
                
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code != 200:
                    error_msg = f"GPT-OSS API 오류 (상태 코드: {response.status_code}): {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                
                result = response.json()
                logger.info("GPT-OSS 응답 수신 완료")
                return result
                
        except httpx.TimeoutException:
            error_msg = f"GPT-OSS API 요청 시간 초과 ({self.timeout}초)"
            logger.error(error_msg)
            raise Exception(error_msg)
        except httpx.RequestError as e:
            error_msg = f"GPT-OSS API 연결 오류: {str(e)}"
            logger.error(error_msg)
            raise Exception(error_msg)
        except Exception as e:
            logger.error(f"GPT-OSS 요청 처리 중 오류: {str(e)}")
            raise
    
    def _build_harmony_prompt(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None
    ) -> str:
        """GPT-OSS Harmony 형식에 맞게 프롬프트를 구성합니다."""
        
        if system_prompt:
            return f"""<|system|>
{system_prompt}
<|user|>
{prompt}
<|assistant|>"""
        else:
            return f"""<|user|>
{prompt}
<|assistant|>"""
    
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

    def _build_rag_prompt(
        self, 
        question: str, 
        context_documents: List[str]
    ) -> str:
        """RAG용 프롬프트를 구성합니다."""
        
        # 컨텍스트 문서 구성
        context_text = "\n\n".join([
            f"[참조 문서 {i+1}]\n{doc}" 
            for i, doc in enumerate(context_documents)
        ])
        
        return f"""참조 문서들:
{context_text}

질문: {question}

위 참조 문서들을 바탕으로 질문에 답변해주세요."""
    
    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """일반적인 텍스트 생성."""
        
        try:
            response = await self._make_request(
                prompt=prompt,
                system_prompt=system_prompt,
                **kwargs
            )
            
            generated_text = response.get("response", "").strip()
            if not generated_text:
                raise Exception("GPT-OSS에서 빈 응답을 받았습니다")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"텍스트 생성 중 오류: {str(e)}")
            raise
    
    async def generate_rag_answer(
        self,
        question: str,
        context_documents: List[str],
        **kwargs
    ) -> str:
        """RAG 시스템용 답변 생성."""
        
        if not context_documents:
            raise ValueError("참조 문서가 제공되지 않았습니다")
        
        system_prompt = self._build_rag_system_prompt()
        rag_prompt = self._build_rag_prompt(question, context_documents)
        
        try:
            logger.info(f"RAG 답변 생성 시작 - 질문: {question[:50]}...")
            logger.info(f"참조 문서 수: {len(context_documents)}")
            
            answer = await self.generate_text(
                prompt=rag_prompt,
                system_prompt=system_prompt,
                **kwargs
            )
            
            logger.info("RAG 답변 생성 완료")
            return answer
            
        except Exception as e:
            logger.error(f"RAG 답변 생성 중 오류: {str(e)}")
            raise
    
    async def check_health(self) -> bool:
        """Ollama 서비스 상태를 확인합니다."""
        
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                
                if response.status_code == 200:
                    tags_data = response.json()
                    models = [model.get("name", "") for model in tags_data.get("models", [])]
                    
                    # 설정된 모델이 사용 가능한지 확인
                    model_available = any(self.model in model for model in models)
                    
                    if model_available:
                        logger.info(f"GPT-OSS 서비스 정상 - 모델 '{self.model}' 사용 가능")
                        return True
                    else:
                        logger.warning(f"모델 '{self.model}'을 찾을 수 없습니다. 사용 가능한 모델: {models}")
                        return False
                        
                return False
                
        except Exception as e:
            logger.error(f"GPT-OSS 상태 확인 중 오류: {str(e)}")
            return False
    
    async def pull_model(self) -> bool:
        """모델을 다운로드합니다."""
        
        try:
            logger.info(f"모델 '{self.model}' 다운로드 시작...")
            
            async with httpx.AsyncClient(timeout=3600) as client:  # 1시간 타임아웃
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": self.model},
                    headers={"Content-Type": "application/json"}
                )
                
                if response.status_code == 200:
                    logger.info(f"모델 '{self.model}' 다운로드 완료")
                    return True
                else:
                    logger.error(f"모델 다운로드 실패: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"모델 다운로드 중 오류: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, any]:
        """현재 설정된 모델 정보를 반환합니다."""
        
        return {
            "base_url": self.base_url,
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "reasoning_level": self.reasoning_level,
            "timeout": self.timeout
        }