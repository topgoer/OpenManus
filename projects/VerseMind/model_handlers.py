"""模型处理器"""

import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from .model_types import ModelType
from projects.ai_poetry_contest.exceptions import ModelConfigError, ModelInitializationError
from app.config import config, LLMSettings
from projects.ai_poetry_contest.poetry_llm import PoetryLLM
import aiohttp
from dataclasses import dataclass
from projects.ai_poetry_contest.model_config import ModelConfig

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelHandler:
    """模型处理器类"""
    
    def __init__(self):
        """初始化模型处理器"""
        self.models: Dict[str, Dict] = {}
        self.model_config = {
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 1000
        }
        logger.info("Initialized ModelHandler")

    async def initialize(self):
        """初始化模型"""
        logger.info("Initializing model...")
        # TODO: 实现实际的模型初始化逻辑

    async def cleanup(self):
        """清理资源"""
        logger.info("Cleaning up model resources...")
        # TODO: 实现实际的资源清理逻辑

    def register_model(self, name: str, config: Dict) -> None:
        """注册模型
        
        Args:
            name: 模型名称
            config: 模型配置
        """
        self.models[name] = config
        
    def get_model_config(self, name: str) -> Optional[Dict]:
        """获取模型配置
        
        Args:
            name: 模型名称
            
        Returns:
            Optional[Dict]: 模型配置，如果不存在则返回 None
        """
        return self.models.get(name)
        
    def get_model_type(self, name: str) -> Optional[ModelType]:
        """获取模型类型
        
        Args:
            name: 模型名称
            
        Returns:
            Optional[ModelType]: 模型类型，如果不存在则返回 None
        """
        config = self.get_model_config(name)
        if not config:
            return None
            
        api_type = config.get("api_type", "ollama")
        return ModelType(api_type)

    def create_model(self, name: str, model_type: ModelType, model_config: ModelConfig) -> Optional[PoetryLLM]:
        """创建模型实例
        
        Args:
            name: 模型名称
            model_type: 模型类型
            model_config: 模型配置
            
        Returns:
            Optional[PoetryLLM]: 创建的模型实例，如果失败则返回 None
        """
        try:
            if name in self.models:
                logger.warning(f"Model {name} already exists")
                return self.models[name]
                
            model = PoetryLLM(
                name=name,
                model_type=model_type,
                model_config=model_config
            )
            
            self.models[name] = model
            logger.info(f"Created model: {name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model {name}: {e}")
            return None

    def get_model(self, name: str) -> Optional[PoetryLLM]:
        """获取模型实例
        
        Args:
            name: 模型名称
            
        Returns:
            Optional[PoetryLLM]: 模型实例，如果不存在则返回 None
        """
        return self.models.get(name)

    def get_all_models(self) -> List[PoetryLLM]:
        """获取所有模型实例
        
        Returns:
            List[PoetryLLM]: 模型实例列表
        """
        return list(self.models.values())

    def remove_model(self, name: str) -> bool:
        """移除模型实例
        
        Args:
            name: 模型名称
            
        Returns:
            bool: 是否成功移除
        """
        if name in self.models:
            del self.models[name]
            logger.info(f"Removed model: {name}")
            return True
        return False

    async def create_poem(self, prompt: str) -> Optional[str]:
        """创建诗歌
        
        Args:
            prompt: 创作提示词
            
        Returns:
            Optional[str]: 创作的诗歌，如果失败则返回 None
        """
        try:
            return await self.llm.create_poem(prompt)
        except Exception as e:
            logger.error(f"创建诗歌失败: {e}")
            return None
            
    async def evaluate_poem(self, poems: List[str], prompt: str) -> Optional[Dict]:
        """评估诗歌
        
        Args:
            poems: 诗歌列表
            prompt: 评分提示词
            
        Returns:
            Optional[Dict]: 评估结果，如果失败则返回 None
        """
        try:
            if not poems:
                logger.warning("没有需要评估的诗歌")
                return None
                
            return await self.llm.evaluate_poem(poems[0], prompt)
        except Exception as e:
            logger.error(f"评估诗歌失败: {e}")
            return None

    async def run(self, prompt: str) -> str:
        """运行 Manus agent"""
        messages = [{"role": "user", "content": prompt}]
        return await self.llm.ask(messages)

    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """解析响应"""
        try:
            # 尝试解析 JSON 响应
            return json.loads(response)
        except json.JSONDecodeError:
            # 如果不是 JSON 格式，返回原始文本
            return {"text": response}
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return None

    async def _create_poem_api(self, prompt: str) -> Optional[str]:
        """使用 API 创建诗歌"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": "你是一个专业的中国诗歌创作者。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "stream": True  # 启用流式输出
                }
                
                async with session.post(
                    f"{self.config.base_url}/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        collected_messages = []
                        async for line in response.content:
                            if line:
                                try:
                                    line = line.decode('utf-8').strip()
                                    if line.startswith('data: '):
                                        line = line[6:]  # 移除 'data: ' 前缀
                                    if line == '[DONE]':
                                        break
                                    chunk = json.loads(line)
                                    if chunk.get('choices') and chunk['choices'][0].get('delta', {}).get('content'):
                                        collected_messages.append(chunk['choices'][0]['delta']['content'])
                                except json.JSONDecodeError:
                                    continue
                        return ''.join(collected_messages)
                    else:
                        error_text = await response.text()
                        logger.error(f"API error: {error_text}")
            return None
        except Exception as e:
            logger.error(f"Error in API call: {e}")
            return None

    async def _evaluate_poem_api(self, poems: List[str], prompt: str) -> Optional[Dict]:
        """使用 API 评估诗歌"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": "你是一个专业的诗歌评论家。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                    "stream": True  # 启用流式输出
                }
                
                async with session.post(
                    f"{self.config.base_url}/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        collected_messages = []
                        async for line in response.content:
                            if line:
                                try:
                                    line = line.decode('utf-8').strip()
                                    if line.startswith('data: '):
                                        line = line[6:]  # 移除 'data: ' 前缀
                                    if line == '[DONE]':
                                        break
                                    chunk = json.loads(line)
                                    if chunk.get('choices') and chunk['choices'][0].get('delta', {}).get('content'):
                                        collected_messages.append(chunk['choices'][0]['delta']['content'])
                                except json.JSONDecodeError:
                                    continue
                        
                        # 尝试解析完整的响应
                        full_response = ''.join(collected_messages)
                        try:
                            # 尝试从文本中提取 JSON 部分
                            start = full_response.find("{")
                            end = full_response.rfind("}") + 1
                            if start >= 0 and end > start:
                                json_str = full_response[start:end]
                                return json.loads(json_str)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON response: {e}")
                    else:
                        error_text = await response.text()
                        logger.error(f"API error: {error_text}")
            return None
        except Exception as e:
            logger.error(f"Error in API call: {e}")
            return None

    async def _create_poem_ollama(self, prompt: str) -> Optional[str]:
        """使用 Ollama 创建诗歌"""
        try:
            async with aiohttp.ClientSession() as session:
                data = {
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False
                }
                
                async with session.post(
                    f"{self.config.base_url}/api/generate",
                    json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response")
                    else:
                        error_text = await response.text()
                        logger.error(f"Ollama error: {error_text}")
            return None
        except Exception as e:
            logger.error(f"Error in Ollama call: {e}")
            return None

    async def _evaluate_poem_ollama(self, poems: List[str], prompt: str) -> Optional[Dict]:
        """使用 Ollama 评估诗歌"""
        try:
            async with aiohttp.ClientSession() as session:
                data = {
                    "model": self.config.model,
                    "prompt": prompt,
                    "stream": False
                }
                
                async with session.post(
                    f"{self.config.base_url}/api/generate",
                    json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        try:
                            # 尝试解析返回的 JSON 字符串
                            response_text = result.get("response", "")
                            if isinstance(response_text, str):
                                # 尝试从文本中提取 JSON 部分
                                start = response_text.find("{")
                                end = response_text.rfind("}") + 1
                                if start >= 0 and end > start:
                                    json_str = response_text[start:end]
                                    return json.loads(json_str)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse JSON response: {e}")
                    else:
                        error_text = await response.text()
                        logger.error(f"Ollama error: {error_text}")
            return None
        except Exception as e:
            logger.error(f"Error in Ollama call: {e}")
            return None

    async def _create_poem_openai(self, prompt: str) -> Optional[str]:
        """使用 OpenAI 创建诗歌"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": "你是一个专业的中国诗歌创作者。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens
                }
                
                async with session.post(
                    f"{self.config.base_url}/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "choices" in result and len(result["choices"]) > 0:
                            return result["choices"][0]["message"]["content"]
                    else:
                        error_text = await response.text()
                        logger.error(f"OpenAI error: {error_text}")
            return None
        except Exception as e:
            logger.error(f"Error in OpenAI call: {e}")
            return None

    async def _evaluate_poem_openai(self, poems: List[str], prompt: str) -> Optional[Dict]:
        """使用 OpenAI 评估诗歌"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": "你是一个专业的诗歌评论家。"},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens
                }
                
                async with session.post(
                    f"{self.config.base_url}/chat/completions",
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if "choices" in result and len(result["choices"]) > 0:
                            try:
                                # 尝试解析返回的 JSON 字符串
                                response_text = result["choices"][0]["message"]["content"]
                                if isinstance(response_text, str):
                                    # 尝试从文本中提取 JSON 部分
                                    start = response_text.find("{")
                                    end = response_text.rfind("}") + 1
                                    if start >= 0 and end > start:
                                        json_str = response_text[start:end]
                                        return json.loads(json_str)
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse JSON response: {e}")
                    else:
                        error_text = await response.text()
                        logger.error(f"OpenAI error: {error_text}")
            return None
        except Exception as e:
            logger.error(f"Error in OpenAI call: {e}")
            return None

    async def generate(self, prompt: str) -> Optional[str]:
        """生成文本
        
        Args:
            prompt: 提示词
            
        Returns:
            Optional[str]: 生成的文本，如果失败则返回 None
        """
        try:
            # 这里可以添加具体的生成逻辑
            # 暂时返回一个示例响应
            return f"这是 {self.model_name} 的示例响应"
        except Exception as e:
            logger.error(f"生成文本失败: {e}")
            return None 