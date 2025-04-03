"""模型类型定义"""

from enum import Enum


class ModelType(Enum):
    """模型类型"""

    CHAT = "chat"
    COMPLETION = "completion"
    API = "api"
    OLLAMA = "ollama"
    OPENAI = "openai"
    DEEPSEEK = "deepseek"
