import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path
import json
from typing import List, Dict, Tuple, Any
import numpy as np
from datetime import datetime
import os

# 默认配置
DEFAULT_CONFIG = {
    "data_dir": "data",  # 相对于项目目录的路径
    "vector_db_dir": "vector_db",  # 相对于 data_dir 的路径
    "model_name": "shibing624/text2vec-base-chinese"
}

class PoetryVectorDB:
    """诗歌向量数据库类，用于存储和检索诗歌的向量表示"""
    
    def __init__(self, collection_name: str = "poetry_collection", config: Dict = None):
        """初始化向量数据库
        
        Args:
            collection_name (str): 集合名称
            config (Dict, optional): 配置信息，包含数据目录等设置
        """
        # 合并配置
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
            
        # 设置数据目录
        self.data_dir = Path(self.config["data_dir"])
        self.data_dir.mkdir(exist_ok=True)
        
        # 初始化持久化客户端
        vector_db_path = self.data_dir / self.config["vector_db_dir"]
        self.client = chromadb.PersistentClient(path=str(vector_db_path))
        
        # 初始化中文文本向量化模型
        self.model = SentenceTransformer(self.config["model_name"])
        
        # 创建或获取集合
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "中国古典诗歌向量数据库"}
        )
        
        # 加载现有数据
        self._load_existing_data()