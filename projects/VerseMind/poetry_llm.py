"""诗歌LLM类"""

import asyncio
import functools
import inspect
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx


# 禁用 httpx 的日志输出
logging.getLogger("httpx").setLevel(logging.WARNING)

# 添加 OpenManus 目录到 Python 路径
open_manus_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, open_manus_dir)

from logger_config import get_logger

from app.config import LLMSettings, config
from app.llm import LLM
from app.schema import Message


# 获取日志记录器
logger = get_logger(__name__)


def get_available_models() -> List[Tuple[str, str]]:
    """获取所有可用的模型配置

    Returns:
        List[Tuple[str, str]]: 模型列表，每个元素为 (模型名称, 模型类型)
    """
    models = []

    # 遍历所有配置
    for name, settings in config.llm.items():
        # 检查配置是否被注释或是否为默认配置
        if name.startswith("#") or name == "default":
            continue

        if isinstance(settings, LLMSettings):
            # 使用配置名称作为模型名称
            models.append((name, settings.api_type))
            logger.info(f"添加模型: {name}，API类型: {settings.api_type}")

    return models


def debug_log(func):
    """装饰器函数，用于包装日志记录方法"""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        frame = inspect.currentframe().f_back
        filename = os.path.basename(frame.f_code.co_filename)
        line_no = frame.f_lineno
        return func(self, f"{args[0]} [{filename}:{line_no}]", *args[1:], **kwargs)

    return wrapper


class PoetryLLM(LLM):
    """诗歌LLM类，继承自LLM基类"""

    # 类变量，用于跟踪请求计数
    _last_request_time = 0  # 添加时间戳跟踪
    _creation_request_count = 0  # 创作阶段请求计数器
    _evaluation_request_count = 0  # 评论阶段请求计数器
    _show_prompts = True  # 是否显示提示词
    _current_stage = "创作"  # 当前阶段，可以是"创作"或"评论"

    def __init__(self, config_name: str, llm_config: Optional[LLMSettings] = None):
        """初始化诗歌LLM

        Args:
            config_name: 配置名称
            llm_config: LLM配置，如果为None则从全局配置中获取
        """
        super().__init__(config_name=config_name, llm_config=llm_config)
        self._last_request_time = 0
        self._response_cache = {}  # 添加响应缓存属性

    @classmethod
    def set_show_prompts(cls, show: bool) -> None:
        """设置是否显示提示词

        Args:
            show: 是否显示提示词
        """
        cls._show_prompts = show
        logger.info(f"提示词显示已{'启用' if show else '禁用'}")

    @classmethod
    def is_showing_prompts(cls) -> bool:
        """获取是否显示提示词

        Returns:
            bool: 是否显示提示词
        """
        return cls._show_prompts

    @classmethod
    def create_poets(cls) -> List[Dict[str, str]]:
        """创建诗人列表

        Returns:
            List[Dict[str, str]]: 诗人列表，每个诗人包含 name 和 model_name
        """
        # 获取可用模型列表（已排序）
        available_models = get_available_models()
        if not available_models:
            logger.error("未找到可用的模型")
            return []

        # 创建诗人列表
        poets = []
        for i, (model_name, api_type) in enumerate(available_models, 1):
            # 创建诗人信息，包含诗人名字和模型名称
            poet = {
                "name": f"诗人{i}",  # 诗人名字
                "model_name": model_name,  # 模型名称
                "api_type": api_type,  # API类型
                "index": i,  # 添加索引
            }
            poets.append(poet)

        return poets

    async def _send_request(self, data: Dict[str, Any], is_stream: bool = False) -> str:
        """发送请求到模型服务器

        Args:
            data: 请求数据
            is_stream: 是否使用流式响应

        Returns:
            str: 模型响应
        """
        # 检查请求类型并增加相应的计数器
        data.get("prompt", "")

        # 根据当前阶段获取请求计数并增加计数器
        if PoetryLLM._current_stage == "评论":
            request_count = PoetryLLM._evaluation_request_count
            request_stage = "评论"
            # 增加评论请求计数器
            PoetryLLM._evaluation_request_count += 1
        else:
            request_count = PoetryLLM._creation_request_count
            request_stage = "创作"
            # 增加创作请求计数器
            PoetryLLM._creation_request_count += 1

        # 不再在这里显示模型信息，避免重复
        # logger.info(f"🤖 正在使用模型 {self.model} 进行{request_stage}... (第{request_count + 1}次请求)")

        # 添加请求间隔检查
        current_time = time.time()
        if current_time - self._last_request_time < 1:  # 1秒间隔
            await asyncio.sleep(1)
        self._last_request_time = current_time

        try:
            # 从配置中获取URL
            base_url = self.base_url if self.base_url else "http://localhost:11434"
            api_endpoint = f"{base_url}/api/generate"

            # 记录请求信息
            logger.debug(
                f"[PoetryLLM] HTTP Request: POST {api_endpoint} [{request_stage}阶段 第{request_count}次请求]"
            )

            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(
                        api_endpoint, json=data, timeout=120.0  # 增加到 120 秒
                    )
                    response.raise_for_status()

                    # 记录响应内容，帮助调试
                    response_text = response.text
                    logger.debug(
                        f"[PoetryLLM] 响应内容: {response_text[:200]}..."
                    )  # 只记录前200个字符

                    # 检查是否是流式响应
                    if data.get("stream", False):
                        # 流式响应，返回原始响应文本
                        return response_text
                    else:
                        # 非流式响应，尝试解析为 JSON
                        try:
                            return response.json()
                        except json.JSONDecodeError as e:
                            logger.error(
                                f"[PoetryLLM] JSON解析错误: {str(e)}, 响应内容: {response_text[:200]}..."
                            )
                            raise
                except httpx.ConnectError as e:
                    logger.error(
                        f"[PoetryLLM] 连接错误: {str(e)}, 请检查服务器是否正在运行"
                    )
                    raise
                except httpx.TimeoutException as e:
                    logger.error(f"[PoetryLLM] 请求超时: {str(e)}, 请检查网络连接")
                    raise
                except httpx.HTTPStatusError as e:
                    logger.error(
                        f"[PoetryLLM] HTTP状态错误: {str(e)}, 状态码: {e.response.status_code}"
                    )
                    raise
                except Exception as e:
                    logger.error(
                        f"[PoetryLLM] 请求错误: {str(e)}, 类型: {type(e).__name__}"
                    )
                    raise
        except Exception as e:
            logger.error(f"[PoetryLLM] 发送请求失败: {str(e)}")
            raise

    async def create_poem(
        self, theme: str, prompt: Optional[str] = None
    ) -> Optional[str]:
        """创建诗歌

        Args:
            theme: 主题
            prompt: 提示词

        Returns:
            Optional[str]: 诗歌内容
        """
        try:
            # 设置当前阶段为创作
            PoetryLLM._current_stage = "创作"

            # 读取参考内容文件
            reference_file = Path(__file__).parent / "prompts" / "reference_content.txt"
            with open(reference_file, "r", encoding="utf-8") as f:
                reference_content = f.read()

            # 如果没有提供提示词，从文件读取
            if not prompt:
                # 读取创作提示词模板
                prompt_file = Path(__file__).parent / "prompts" / "creation_prompt.txt"
                with open(prompt_file, "r", encoding="utf-8") as f:
                    prompt_template = f.read()

                # 将参考内容添加到提示词中
                prompt = f"{prompt_template}\n\n参考内容：\n{reference_content}"

                # 只在计数器为0时显示提示词（第一次请求）
                if PoetryLLM._creation_request_count == 0:
                    logger.info("=== 创作提示词 ===")
                    logger.info(prompt_template)
                    logger.info("==================")
            else:
                # 如果提供了提示词，确保它包含参考内容
                if "参考内容：" not in prompt:
                    prompt = f"{prompt}\n\n参考内容：\n{reference_content}"

                # 只在计数器为0时显示提示词（第一次请求）
                if PoetryLLM._creation_request_count == 0:
                    logger.info("=== 创作提示词 ===")
                    logger.info(prompt)
                    logger.info("==================")

            # 记录完整的提示词
            logger.debug(f"[PoetryLLM] 完整创作提示词: {prompt}")

            # 使用基类的 ask 方法
            try:
                response = await self.ask([{"role": "user", "content": prompt}])
            except Exception as e:
                logger.error(f"[PoetryLLM] 发送创作请求失败: {str(e)}")
                # 尝试使用非流式请求
                logger.info("[PoetryLLM] 尝试使用非流式请求...")
                response = await self.ask(
                    [{"role": "user", "content": prompt}], stream=False
                )

            # 记录原始响应
            logger.debug(f"[PoetryLLM] 原始创作响应: {response}")

            # 清理响应，只保留标题和创作思路说明
            lines = response.split("\n")
            cleaned_lines = []
            for line in lines:
                if line.startswith("## "):  # 标题
                    cleaned_lines.append(line)
                elif line.startswith("创作思路说明："):
                    cleaned_lines.append(line)
                elif line.strip():  # 诗歌内容，包括思考过程
                    cleaned_lines.append(line)

            # 创作阶段结束，记录请求数
            logger.info(
                f"[PoetryLLM] 创作阶段完成，共发送 {PoetryLLM._creation_request_count} 个请求"
            )

            result = "\n".join(cleaned_lines)
            logger.debug(f"[PoetryLLM] 清理后的创作响应: {result}")

            return result

        except Exception as e:
            logger.error(f"创建诗歌失败: {e}")
            return None

    async def evaluate_poem(
        self,
        poem: str,
        theme: str,
        is_human_poem: bool = False,
        prompt: Optional[str] = None,
    ) -> Optional[Dict]:
        """评估诗歌

        Args:
            poem: 诗歌内容
            theme: 主题
            is_human_poem: 是否是人类诗歌
            prompt: 评分提示词，如果提供则使用此提示词

        Returns:
            Optional[Dict]: 评估结果，如果评论失败则返回None
        """
        try:
            # 设置当前阶段为评论
            PoetryLLM._current_stage = "评论"

            # 清理诗歌内容，只保留标题和诗歌，不包含思考过程
            cleaned_poem = self._clean_poem_for_evaluation(poem)

            # 如果提供了提示词，直接使用
            if prompt:
                # 在评论阶段，每首诗都显示提示词
                logger.info("=== 评分提示词 ===")
                logger.info(prompt)  # 记录提供的提示词
                logger.info("==================")

                # 记录完整的提示词
                logger.debug(f"[PoetryLLM] 完整评分提示词: {prompt}")

                # 使用基类的 ask 方法
                try:
                    response = await self.ask([{"role": "user", "content": prompt}])
                except Exception as e:
                    logger.error(f"[PoetryLLM] 发送评分请求失败: {str(e)}")
                    # 尝试使用非流式请求
                    logger.info("[PoetryLLM] 尝试使用非流式请求...")
                    response = await self.ask(
                        [{"role": "user", "content": prompt}], stream=False
                    )
            else:
                # 读取参考内容文件
                reference_file = (
                    Path(__file__).parent / "prompts" / "reference_content.txt"
                )
                with open(reference_file, "r", encoding="utf-8") as f:
                    reference_content = f.read()

                # 从文件读取提示词
                prompt_file = Path(__file__).parent / "prompts" / "scoring_prompt.txt"
                with open(prompt_file, "r", encoding="utf-8") as f:
                    prompt_template = f.read()

                # 格式化提示词，添加人类诗歌标记
                poem_type = "人类诗歌" if is_human_poem else "AI诗歌"
                prompt = prompt_template.replace("{theme}", reference_content).replace(
                    "{poem}", cleaned_poem
                )
                prompt = f"{prompt}\n\n注意：这是一首{poem_type}，请根据诗歌类型进行适当评价。"

                # 在评论阶段，每首诗都显示提示词
                logger.info("=== 评分提示词 ===")
                logger.info(prompt)  # 记录替换后的提示词
                logger.info("==================")

                # 记录完整的提示词
                logger.debug(f"[PoetryLLM] 完整评分提示词: {prompt}")

                # 使用基类的 ask 方法
                try:
                    response = await self.ask([{"role": "user", "content": prompt}])
                except Exception as e:
                    logger.error(f"[PoetryLLM] 发送评分请求失败: {str(e)}")
                    # 尝试使用非流式请求
                    logger.info("[PoetryLLM] 尝试使用非流式请求...")
                    response = await self.ask(
                        [{"role": "user", "content": prompt}], stream=False
                    )

            # 记录原始响应
            logger.debug(f"[PoetryLLM] 原始评分响应: {response}")

            # 解析评分结果
            try:
                # 尝试从响应中提取评分部分
                score_match = re.search(r"总分：(\d+)", response)
                if not score_match:
                    # 尝试其他可能的格式
                    score_match = re.search(r"总评分：(\d+)", response)
                    if not score_match:
                        # 尝试从表格中提取分数并计算总分
                        logger.warning("[PoetryLLM] 未找到总分，尝试从表格中提取分数")
                        # 查找所有可能的分数
                        score_matches = re.findall(r"(\d+\.?\d*)\s*分", response)
                        if score_matches:
                            # 将找到的分数转换为浮点数并计算平均值
                            scores = [float(score) for score in score_matches]
                            total_score = int(
                                sum(scores) / len(scores) * 10
                            )  # 转换为60分制
                            logger.info(
                                f"[PoetryLLM] 从表格中提取的分数: {scores}, 计算得到总分: {total_score}"
                            )
                        else:
                            # 如果仍然找不到分数，设置总分为None，但仍然保留评论
                            logger.warning(
                                "[PoetryLLM] 无法提取分数，评论有效但不参与排名"
                            )
                            total_score = None
                    else:
                        total_score = int(score_match.group(1))
                else:
                    total_score = int(score_match.group(1))

                # 提取维度评分
                dimensions = {}
                dimension_pattern = r"(\w+)：(\d+)"
                for match in re.finditer(dimension_pattern, response):
                    dimension, score = match.groups()
                    dimensions[dimension] = int(score)

                # 如果维度评分为空，尝试从表格中提取
                if not dimensions:
                    # 尝试从表格中提取维度评分
                    table_pattern = r"\|.*?(\w+).*?\|.*?(\d+\.?\d*).*?\|"
                    for match in re.finditer(table_pattern, response):
                        dimension, score = match.groups()
                        dimensions[dimension] = int(float(score))

                # 如果仍然没有维度评分，使用空字典，但仍然保留评论
                if not dimensions:
                    logger.warning("[PoetryLLM] 无法提取维度评分，评论有效但不参与排名")
                    dimensions = {}

                # 提取点评
                comment_match = re.search(r"点评：(.+?)(?=\n|$)", response)
                comment = comment_match.group(1) if comment_match else "无"

                # 如果点评为空，尝试提取其他可能的点评内容
                if comment == "无":
                    # 尝试查找可能的点评内容
                    comment_patterns = [
                        r"评语：(.+?)(?=\n|$)",
                        r"评价：(.+?)(?=\n|$)",
                        r"点评：(.+?)(?=\n|$)",
                        r"总结：(.+?)(?=\n|$)",
                    ]
                    for pattern in comment_patterns:
                        comment_match = re.search(pattern, response)
                        if comment_match:
                            comment = comment_match.group(1)
                            break

                    # 如果仍然没有找到点评，尝试提取整个响应作为点评
                    if comment == "无":
                        # 尝试提取整个响应作为点评
                        comment = response.strip()
                        if len(comment) > 500:  # 如果响应太长，截取前500个字符
                            comment = comment[:500] + "..."
                        logger.info("[PoetryLLM] 使用整个响应作为点评")

                # 评论阶段结束，记录请求数
                logger.info(
                    f"[PoetryLLM] 评论阶段完成，共发送 {PoetryLLM._evaluation_request_count} 个请求"
                )

                result = {
                    "total_score": total_score,
                    "dimensions": dimensions,
                    "comment": comment,
                }

                # 如果是人类诗歌，记录特殊日志
                if is_human_poem:
                    logger.info(
                        f"[PoetryLLM] 已完成对人类诗歌的评价，总分: {total_score}"
                    )

                logger.debug(f"[PoetryLLM] 解析后的评分结果: {result}")

                return result

            except Exception as e:
                logger.error(f"解析评分结果失败: {e}")
                # 返回None，表示评论无效
                return None

        except Exception as e:
            logger.error(f"评估诗歌失败: {e}")
            # 返回None，表示评论无效
            return None

    def _clean_poem_for_evaluation(self, poem: str) -> str:
        """清理诗歌内容，只保留标题和诗歌，不包含思考过程

        Args:
            poem: 原始诗歌内容

        Returns:
            str: 清理后的诗歌内容
        """
        lines = poem.split("\n")
        cleaned_lines = []
        in_poem = False

        for line in lines:
            # 保留标题
            if line.startswith("## "):
                cleaned_lines.append(line)
                in_poem = True
                continue

            # 跳过创作思路说明
            if line.startswith("创作思路说明："):
                in_poem = False
                continue

            # 跳过思考过程
            if line.startswith("<think>") or line.startswith("</think>"):
                in_poem = False
                continue

            # 如果是诗歌内容，保留
            if in_poem and line.strip():
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    @debug_log
    def debug(self, msg: str) -> None:
        """调试日志"""
        logger.debug(msg)

    @debug_log
    def info(self, msg: str) -> None:
        """信息日志"""
        logger.info(msg)

    @debug_log
    def warning(self, msg: str) -> None:
        """警告日志"""
        logger.warning(msg)

    @debug_log
    def error(self, msg: str) -> None:
        """错误日志"""
        logger.error(msg)

    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """重写父类的 ask 方法以支持 Ollama 的 /api/generate 端点"""
        try:
            # 显示当前工作的诗人
            stage = "评论" if PoetryLLM._current_stage == "评论" else "创作"

            # 获取当前请求计数
            if stage == "评论":
                request_count = PoetryLLM._evaluation_request_count
                # 增加计数器
                PoetryLLM._evaluation_request_count += 1
            else:
                request_count = PoetryLLM._creation_request_count
                # 增加计数器
                PoetryLLM._creation_request_count += 1

            # 在这里显示模型信息，确保在评论阶段也能看到
            print(
                f"\n🤖 正在使用模型 {self.model} 进行{stage}... (第{request_count + 1}次请求)\n"
            )

            # 如果是 Ollama 模型，使用 /api/generate 端点
            if self.api_type == "ollama":
                # 合并消息
                content = ""
                for msg in messages:
                    if isinstance(msg, dict):
                        content += f"{msg['role']}: {msg['content']}\n\n"
                    else:
                        content += f"{msg.role}: {msg.content}\n\n"

                # 添加系统消息
                if system_msgs:
                    for msg in system_msgs:
                        if isinstance(msg, dict):
                            content = f"{msg['role']}: {msg['content']}\n\n" + content
                        else:
                            content = f"{msg.role}: {msg.content}\n\n" + content

                # 检查缓存
                cache_key = f"{self.model}:{content}:{stream}:{temperature}"
                if cache_key in self._response_cache:
                    logger.debug("[PoetryLLM] 使用缓存的响应")
                    return self._response_cache[cache_key]

                # 准备请求参数
                params = {
                    "model": self.model,
                    "prompt": content,
                    "stream": stream,
                    "temperature": (
                        temperature if temperature is not None else self.temperature
                    ),
                    "max_tokens": self.max_tokens,
                }

                # 记录请求参数
                logger.debug(
                    f"[PoetryLLM] 请求参数: {json.dumps(params, ensure_ascii=False)}"
                )

                # 使用我们自己的 _send_request 方法，它会处理请求计数
                response_text = await self._send_request(params)

                if stream:
                    # 处理流式响应
                    collected_messages = []

                    # 使用当前阶段
                    stage = PoetryLLM._current_stage

                    # 不再输出重复的日志
                    # logger.info(f"[PoetryLLM] {self.model} 正在{stage}...")  # 开始提示

                    # 解析流式响应
                    lines = response_text.strip().split("\n")
                    for line in lines:
                        if line.strip():
                            try:
                                chunk = json.loads(line)
                                if "response" in chunk:
                                    response_text = chunk["response"]
                                    collected_messages.append(response_text)
                                    # 直接打印响应，保持自然速度
                                    print(response_text, end="", flush=True)
                            except json.JSONDecodeError as e:
                                logger.error(
                                    f"[PoetryLLM] JSON解析错误: {str(e)}, 行: {line}"
                                )
                                continue

                    print("\n")  # 添加额外的换行
                    result = "".join(collected_messages).strip()
                else:
                    # 处理非流式响应
                    if isinstance(response_text, str):
                        # 如果返回的是字符串，尝试解析为 JSON
                        try:
                            response_json = json.loads(response_text)
                            if "response" in response_json:
                                result = response_json["response"].strip()
                            else:
                                result = response_text.strip()
                        except json.JSONDecodeError:
                            # 如果不是有效的 JSON，直接使用字符串
                            result = response_text.strip()
                    elif (
                        isinstance(response_text, dict) and "response" in response_text
                    ):
                        result = response_text["response"].strip()
                    else:
                        logger.error(f"[PoetryLLM] 非流式响应格式错误: {response_text}")
                        result = ""

                # 缓存响应
                self._response_cache[cache_key] = result

                return result
            else:
                # 对于非Ollama模型，使用父类的ask方法
                return await super().ask(messages, system_msgs, stream, temperature)

        except Exception as e:
            logger.error(f"[PoetryLLM] ask方法失败: {e}")
            raise

    @classmethod
    def reset_counters(cls) -> None:
        """重置所有计数器，在整个比赛结束后调用"""
        logger.info(
            f"[PoetryLLM] 当前请求计数 - 创作阶段: {cls._creation_request_count}, 评论阶段: {cls._evaluation_request_count}"
        )
        cls._creation_request_count = 0
        cls._evaluation_request_count = 0
        cls._current_stage = "创作"  # 重置当前阶段为创作
        logger.info("[PoetryLLM] 所有计数器已重置")

    @classmethod
    def configure_logging(cls) -> None:
        """配置日志级别，将特定日志消息的级别改为 DEBUG"""
        # 获取 httpx 的日志记录器
        httpx_logger = logging.getLogger("httpx")
        # 设置为 WARNING 级别，减少 httpx 的日志输出
        httpx_logger.setLevel(logging.WARNING)

        # 获取 app.llm 的日志记录器
        llm_logger = logging.getLogger("app.llm")

        # 创建一个过滤器，将特定消息的级别改为 DEBUG
        class TokenEstimateFilter(logging.Filter):
            def filter(self, record):
                if (
                    "Estimated completion tokens for streaming response"
                    in record.getMessage()
                ):
                    # 将消息级别改为 DEBUG
                    record.levelno = logging.DEBUG
                    record.levelname = "DEBUG"
                    # 返回 False 阻止消息通过
                    return False
                # 对于其他消息，返回 True 允许它们正常通过
                return True

        # 添加过滤器
        llm_logger.addFilter(TokenEstimateFilter())

        # 设置日志级别为 DEBUG
        logger.setLevel(logging.DEBUG)
        logger.debug("[PoetryLLM] 日志级别已配置为 DEBUG")
