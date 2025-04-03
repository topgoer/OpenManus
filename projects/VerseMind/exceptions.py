"""自定义异常类"""

class PoetryContestError(Exception):
    """诗歌比赛基础异常类"""
    pass

class ModelConfigError(PoetryContestError):
    """模型配置错误"""
    pass

class ModelInitializationError(PoetryContestError):
    """模型初始化错误"""
    pass

class PromptError(PoetryContestError):
    """提示词错误"""
    pass

class EvaluationError(PoetryContestError):
    """评估错误"""
    pass 