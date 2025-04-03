"""AI诗歌比赛核心逻辑"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from logger_config import get_logger
from poetry_llm import PoetryLLM
from poetry_vector_db import PoetryVectorDB


# 获取日志记录器
logger = get_logger(__name__)


@dataclass
class Poet:
    """诗人信息"""

    name: str
    model_name: str
    model: PoetryLLM
    poems: List[Dict] = field(default_factory=list)
    evaluations: List[Dict] = field(default_factory=list)


@dataclass
class AIPoetryContest:
    """AI诗歌比赛类"""

    theme: str
    poets: List[Poet]
    results_dir: Path
    reference_content: str = ""
    creation_prompt: str = ""
    scoring_prompt: str = ""
    anonymous_poems: List[Dict] = field(default_factory=list)
    human_poem: Optional[Dict] = None
    vector_db: Optional[PoetryVectorDB] = None

    def __post_init__(self):
        """初始化后的处理"""
        # 创建结果目录
        self.results_dir.mkdir(parents=True, exist_ok=True)
        # 从主题中移除"标题："前缀
        theme_content = self.theme.replace("标题：", "").strip()
        logger.info(f"已初始化比赛，主题: {theme_content}")
        logger.info(f"诗人数量: {len(self.poets)}")

        # 初始化参考内容和提示词
        self.reference_content = ""
        self.creation_prompt = ""
        self.scoring_prompt = ""

        # 初始化向量数据库
        vector_db_path = self.results_dir / "poetry_vectors"
        self.vector_db = PoetryVectorDB(db_path=str(vector_db_path))
        logger.info(f"已初始化诗歌向量数据库，存储路径: {vector_db_path.absolute()}")

    async def run(self) -> bool:
        """运行比赛

        Returns:
            bool: 比赛是否成功完成
        """
        if not self.poets:
            logger.error("没有可用的诗人，比赛无法进行")
            return False

        try:
            # 读取参考内容
            try:
                reference_file = (
                    Path(__file__).parent / "prompts" / "reference_content.txt"
                )
                with open(reference_file, "r", encoding="utf-8") as f:
                    self.reference_content = f.read()

                # 断言参考内容不为空
                assert self.reference_content, "参考内容不能为空"
            except Exception as e:
                logger.error(f"读取参考内容文件失败: {str(e)}")
                return False

            # 读取创作提示词
            try:
                prompt_file = Path(__file__).parent / "prompts" / "creation_prompt.txt"
                with open(prompt_file, "r", encoding="utf-8") as f:
                    self.creation_prompt = f.read()

                # 断言创作提示词不为空
                assert self.creation_prompt, "创作提示词不能为空"
            except Exception as e:
                logger.error(f"读取创作提示词文件失败: {str(e)}")
                return False

            # 读取评分提示词
            try:
                prompt_file = Path(__file__).parent / "prompts" / "scoring_prompt.txt"
                with open(prompt_file, "r", encoding="utf-8") as f:
                    self.scoring_prompt = f.read()

                # 断言评分提示词不为空
                assert self.scoring_prompt, "评分提示词不能为空"
            except Exception as e:
                logger.error(f"读取评分提示词文件失败: {str(e)}")
                return False

            # 创作阶段
            logger.info("开始创作阶段...")
            for poet in self.poets:
                try:
                    # 格式化提示词
                    prompt = self.creation_prompt.replace(
                        "{theme}", self.reference_content
                    )

                    # 断言格式化后的提示词不为空
                    assert prompt, "格式化后的创作提示词不能为空"

                    # 确保提示词包含参考内容
                    if "参考内容：" not in prompt:
                        prompt = f"{prompt}\n\n参考内容：\n{self.reference_content}"

                    # 创建诗歌
                    logger.info(f"{poet.model_name} 正在创作...")
                    poem = await poet.model.create_poem(self.theme, prompt)

                    if poem:
                        # 保存诗歌
                        poem_info = {
                            "id": len(self.anonymous_poems) + 1,
                            "poet": poet.name,
                            "model": poet.model_name,
                            "content": poem,
                        }
                        self.anonymous_poems.append(poem_info)
                        poet.poems.append(poem_info)
                        logger.info(f"{poet.model_name} 创作成功")

                        # 添加到向量数据库
                        metadata = {
                            "poet": poet.name,
                            "model": poet.model_name,
                            "theme": self.theme,
                            "timestamp": datetime.now().isoformat(),
                        }
                        self.vector_db.add_poem(poem, metadata)
                        logger.info(f"已将诗歌添加到向量数据库")
                    else:
                        logger.error(f"{poet.model_name} 创作失败")
                except Exception as e:
                    logger.error(f"{poet.model_name} 创作出错: {e}")
                    continue

            if not self.anonymous_poems:
                logger.error("所有诗人都创作失败，比赛终止")
                return False

            # 评价阶段
            logger.info("开始评价阶段...")

            # 检查是否存在人类诗歌文件，如果存在则添加到评价中
            human_poem_path = self.results_dir / "human_poem_example.txt"
            human_poem_added = False
            if human_poem_path.exists():
                try:
                    # 读取人类诗歌文件
                    with open(human_poem_path, "r", encoding="utf-8") as f:
                        human_poem = f.read()

                    # 断言人类诗歌不为空
                    assert human_poem, "人类诗歌不能为空"

                    # 添加人类诗歌到匿名诗歌文件
                    if self.add_human_poem(human_poem):
                        logger.info("人类诗歌已添加到评价中")
                        human_poem_added = True
                    else:
                        logger.error("添加人类诗歌失败")
                except Exception as e:
                    logger.error(f"添加人类诗歌时出错: {e}")

            # 断言人类诗歌是否成功添加
            if human_poem_path.exists():
                assert human_poem_added, "人类诗歌文件存在但未能成功添加到评价中"

            for poet in self.poets:
                for poem_info in self.anonymous_poems:
                    try:
                        # 断言诗歌内容不为空
                        assert poem_info["content"], "诗歌内容不能为空"

                        # 格式化评分提示词
                        formatted_scoring_prompt = self.scoring_prompt.replace(
                            "{theme}", self.reference_content
                        ).replace("{poem}", poem_info["content"])

                        # 断言格式化后的评分提示词不为空
                        assert formatted_scoring_prompt, "格式化后的评分提示词不能为空"

                        # 评价诗歌
                        logger.info(f"{poet.model_name} 正在评价...")

                        # 检查是否是人类诗歌
                        is_human_poem = poem_info.get("model") == "human"
                        if is_human_poem:
                            logger.info(
                                f"{poet.model_name} 正在评价人类诗歌，ID: {poem_info['id']}"
                            )

                        # 使用格式化后的评分提示词，并设置 is_human_poem 参数
                        evaluation = await poet.model.evaluate_poem(
                            poem_info["content"],
                            self.theme,
                            is_human_poem=is_human_poem,
                            prompt=formatted_scoring_prompt,
                        )

                        if evaluation:
                            # 保存评价结果
                            evaluation_info = {
                                "poem_id": poem_info["id"],
                                "poet": poet.name,
                                "model": poet.model_name,
                                "score": evaluation["total_score"],
                                "dimensions": evaluation["dimensions"],
                                "comment": evaluation["comment"],
                            }
                            poet.evaluations.append(evaluation_info)
                            logger.info(
                                f"{poet.model_name} 完成了对诗歌 {poem_info['id']} 的评价"
                            )
                        else:
                            logger.error(f"{poet.model_name} 评价诗歌 {poem_info['id']} 失败")
                    except Exception as e:
                        logger.error(f"{poet.model_name} 评价诗歌出错: {e}")
                        continue

            # 显示向量数据库统计信息
            if self.vector_db:
                poems_count = len(self.vector_db.get_all_poems())
                db_path = self.vector_db.db_path.absolute()
                logger.info(f"向量数据库统计: 共存储 {poems_count} 首诗歌，存储路径: {db_path}")

                # 显示一些示例查询
                if poems_count > 0:
                    # 使用主题作为查询
                    similar_poems = self.vector_db.search_similar_poems(self.theme, k=3)
                    if similar_poems:
                        logger.info("向量数据库示例查询 - 与主题相似的诗歌:")
                        for i, (poem, similarity) in enumerate(similar_poems):
                            logger.info(
                                f"  相似度 {similarity:.2f}: {poem['content'][:50]}..."
                            )
            else:
                logger.warning("向量数据库未初始化，无法显示统计信息")

            # 保存比赛结果
            try:
                # 保存匿名诗歌
                anonymous_file = self.results_dir / "anonymous_poems.json"
                with open(anonymous_file, "w", encoding="utf-8") as f:
                    json.dump(self.anonymous_poems, f, ensure_ascii=False, indent=2)
                logger.info(f"已创建匿名诗歌文件，共 {len(self.anonymous_poems)} 首诗歌")

                # 断言匿名诗歌文件包含所有诗歌，包括人类诗歌（如果存在）
                with open(anonymous_file, "r", encoding="utf-8") as f:
                    saved_poems = json.load(f)
                assert len(saved_poems) == len(
                    self.anonymous_poems
                ), "保存的匿名诗歌数量与内存中的不一致"
                if human_poem_added:
                    assert any(
                        poem["model"] == "human" for poem in saved_poems
                    ), "保存的匿名诗歌中不包含人类诗歌"
            except Exception as e:
                logger.error(f"创建匿名诗歌文件失败: {e}")
                return False

            # 保存评价结果
            try:
                # 保存每个诗人的评价结果
                for poet in self.poets:
                    if poet.evaluations:
                        evaluation_file = (
                            self.results_dir / f"{poet.name}_evaluations.json"
                        )
                        with open(evaluation_file, "w", encoding="utf-8") as f:
                            json.dump(poet.evaluations, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"保存评价结果失败: {e}")
                return False

            logger.info(f"比赛结果已保存到 {self.results_dir} 目录")
            return True

        except Exception as e:
            logger.error(f"比赛运行出错: {str(e)}")
            return False

    async def _evaluate_poem(self, poem: str, poet: Poet) -> bool:
        """评估单首诗歌

        Args:
            poem: 要评估的诗歌
            poet: 评估的诗人

        Returns:
            bool: 是否评估成功
        """
        try:
            # 读取匿名诗歌文件
            anonymous_file = self.results_dir / "anonymous_poems.json"
            with open(anonymous_file, "r", encoding="utf-8") as f:
                anonymous_poems = json.load(f)

            # 检查是否已经评估过
            poem_id = None
            for p in anonymous_poems:
                if p["content"] == poem:
                    poem_id = p["id"]
                    break

            if not poem_id:
                logger.error(f"未找到诗歌: {poem[:50]}...")
                return False

            # 检查是否已经评估过
            if poem_id in poet.evaluations:
                logger.info(f"诗人 {poet.name} 已经评估过诗歌 {poem_id}")
                return True

            # 获取评分结果
            result = await poet.model.evaluate_poem(poem, self.theme)

            if not result:
                logger.error(f"诗人 {poet.name} 评估诗歌 {poem_id} 失败")
                return False

            # 保存评分结果
            poet.evaluations[poem_id] = result
            logger.info(f"诗人 {poet.name} 评估诗歌 {poem_id} 成功")
            return True

        except Exception as e:
            logger.error(f"评估诗歌时出错: {str(e)}")
            return False

    def announce_results(self) -> bool:
        """宣布比赛结果

        Returns:
            bool: 是否成功宣布结果
        """
        try:
            # 读取匿名诗歌文件
            anonymous_file = self.results_dir / "anonymous_poems.json"
            if not anonymous_file.exists():
                logger.error("匿名诗歌文件不存在")
                return False

            with open(anonymous_file, "r", encoding="utf-8") as f:
                json.load(f)

            # 计算每个诗人的平均得分
            poet_scores = {}
            for poet in self.poets:
                if poet.evaluations:
                    total_score = sum(e["score"] for e in poet.evaluations)
                    average_score = total_score / len(poet.evaluations)
                    poet_scores[poet.name] = average_score

            # 按得分排序
            sorted_poets = sorted(poet_scores.items(), key=lambda x: x[1], reverse=True)

            # 保存排名结果
            ranking_file = self.results_dir / "ranking.json"
            with open(ranking_file, "w", encoding="utf-8") as f:
                json.dump(sorted_poets, f, ensure_ascii=False, indent=2)

            # 显示向量数据库统计信息
            if self.vector_db:
                poems_count = len(self.vector_db.get_all_poems())
                db_path = self.vector_db.db_path.absolute()
                logger.info(f"向量数据库最终统计: 共存储 {poems_count} 首诗歌，存储路径: {db_path}")

                # 显示一些示例查询
                if poems_count > 0:
                    # 使用主题作为查询
                    similar_poems = self.vector_db.search_similar_poems(self.theme, k=3)
                    if similar_poems:
                        logger.info("向量数据库示例查询 - 与主题相似的诗歌:")
                        for i, (poem, similarity) in enumerate(similar_poems):
                            logger.info(
                                f"  相似度 {similarity:.2f}: {poem['content'][:50]}..."
                            )
            else:
                logger.warning("向量数据库未初始化，无法显示统计信息")

            return True

        except Exception as e:
            logger.error(f"宣布比赛结果失败: {str(e)}")
            return False

    def add_human_poem(self, poem_content: str) -> bool:
        """添加人类诗歌到匿名诗歌列表

        Args:
            poem_content: 诗歌内容

        Returns:
            bool: 是否成功添加
        """
        try:
            # 创建人类诗歌信息
            human_poem = {
                "id": len(self.anonymous_poems) + 1,
                "poet": "人类诗人",
                "model": "human",
                "content": poem_content,
            }

            # 添加到匿名诗歌列表
            self.anonymous_poems.append(human_poem)

            # 添加到向量数据库
            if self.vector_db:
                metadata = {
                    "poet": "人类诗人",
                    "model": "human",
                    "theme": self.theme,
                    "timestamp": datetime.now().isoformat(),
                }
                self.vector_db.add_poem(poem_content, metadata)
                logger.info("已将人类诗歌添加到向量数据库")

            # 断言人类诗歌已成功添加到匿名诗歌列表
            assert any(
                poem["model"] == "human" for poem in self.anonymous_poems
            ), "人类诗歌未能成功添加到匿名诗歌列表"

            logger.info(f"人类诗歌已添加到匿名诗歌列表，ID: {human_poem['id']}")
            return True
        except Exception as e:
            logger.error(f"添加人类诗歌失败: {e}")
            return False

    def _save_results(self) -> bool:
        """保存比赛结果

        Returns:
            bool: 是否保存成功
        """
        try:
            # 保存匿名诗歌
            with open(
                self.results_dir / "anonymous_poems.json", "w", encoding="utf-8"
            ) as f:
                json.dump(self.anonymous_poems, f, ensure_ascii=False, indent=2)

            # 保存评分结果
            evaluation_results = []
            for poet in self.poets:
                for evaluation in poet.evaluations:
                    evaluation_results.append(
                        {
                            "poem_id": evaluation["poem_id"],
                            "poet": poet.name,
                            "model": poet.model_name,
                            "score": evaluation["score"],
                            "dimensions": evaluation["dimensions"],
                            "comment": evaluation["comment"],
                        }
                    )

            with open(
                self.results_dir / "evaluation_results.json", "w", encoding="utf-8"
            ) as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

            return True

        except Exception as e:
            logger.error(f"保存结果时出错: {str(e)}")
            return False

    def search_similar_poems(
        self, query: str, k: int = 5
    ) -> List[Tuple[Dict[str, Any], float]]:
        """搜索相似诗歌

        Args:
            query (str): 查询文本
            k (int): 返回结果数量

        Returns:
            List[Tuple[Dict[str, Any], float]]: 相似诗歌列表，每个元素为(诗歌数据, 相似度)
        """
        if not self.vector_db:
            logger.error("向量数据库未初始化")
            return []

        try:
            results = self.vector_db.search_similar_poems(query, k)
            logger.info(f"找到 {len(results)} 首相似诗歌")
            return results
        except Exception as e:
            logger.error(f"搜索相似诗歌时出错: {e}")
            return []
