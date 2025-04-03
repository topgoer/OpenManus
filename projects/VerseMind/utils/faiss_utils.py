"""FAISS工具模块，用于处理FAISS的初始化和配置"""

import io
import logging
from contextlib import redirect_stderr


# 获取日志记录器
logger = logging.getLogger(__name__)

# 临时重定向stderr以抑制FAISS的GPU警告
with redirect_stderr(io.StringIO()):
    import faiss


def get_faiss_index(dimension: int):
    """获取FAISS索引实例

    Args:
        dimension (int): 向量维度

    Returns:
        faiss.IndexFlatL2: FAISS索引实例
    """
    # 创建CPU索引
    cpu_index = faiss.IndexFlatL2(dimension)

    # 尝试使用GPU
    try:
        # 检查是否有可用的GPU资源
        if hasattr(faiss, "StandardGpuResources"):
            try:
                logger.info("正在尝试使用GPU版本的FAISS...")
                # 创建GPU资源
                res = faiss.StandardGpuResources()
                # 将CPU索引转移到GPU
                gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                logger.info("成功启用GPU版本的FAISS！")
                return gpu_index
            except Exception as e:
                logger.warning(f"GPU版本初始化失败: {str(e)}")
                logger.info("回退到CPU版本的FAISS")
    except (AttributeError, ImportError):
        logger.info("未检测到GPU支持，使用CPU版本的FAISS")

    return cpu_index


def save_faiss_index(index, path: str):
    """保存FAISS索引

    Args:
        index: FAISS索引实例
        path (str): 保存路径
    """
    # 如果是GPU索引，先转回CPU
    if hasattr(index, "getDevice"):
        index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, path)


def load_faiss_index(path: str):
    """加载FAISS索引

    Args:
        path (str): 索引文件路径

    Returns:
        faiss.IndexFlatL2: 加载的FAISS索引实例
    """
    # 加载CPU索引
    cpu_index = faiss.read_index(path)

    # 尝试转移到GPU
    try:
        if hasattr(faiss, "StandardGpuResources"):
            try:
                logger.info("正在尝试将索引转移到GPU...")
                res = faiss.StandardGpuResources()
                gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                logger.info("成功将索引转移到GPU！")
                return gpu_index
            except Exception as e:
                logger.warning(f"转移到GPU失败: {str(e)}")
                logger.info("使用CPU版本的索引")
    except (AttributeError, ImportError):
        logger.info("未检测到GPU支持，使用CPU版本的索引")

    return cpu_index
