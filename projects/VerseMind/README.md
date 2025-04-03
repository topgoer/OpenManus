# AI 诗歌比赛系统

在人机共生的创作前沿，诗歌不再只是语言的艺术，而进化为认知革命的演练场——每个隐喻都是打开新维度的脑洞，每次意象碰撞都在重塑人类意识的宇宙边界。

一个基于大语言模型的诗歌创作和评价系统。

## 功能特点

- 支持多个 AI 诗人同时参与创作
- 支持人类诗歌的加入和评价
- 提供公平的匿名评价机制
- 集成向量数据库，支持语义检索
- 自动保存比赛结果和评价数据

## 系统架构

- `poetry_llm.py`: 大语言模型接口封装
- `poetry_contest.py`: 诗歌比赛核心逻辑
- `poetry_vector_db.py`: 向量数据库实现
- `run_contest.py`: 比赛运行入口

## 技术特点

1. **智能创作**
   - 支持多种大语言模型
   - 可配置的创作提示词
   - 灵活的主题设置

2. **公平评价**
   - 匿名评价机制
   - 多维度评分标准
   - 详细评价意见

3. **语义检索**
   - 基于向量数据库的诗歌检索
   - 支持相似度搜索
   - 高效的向量索引

## 安装依赖

```bash
pip install -r requirements.txt
```

## 快速开始

1. 配置环境变量：
```bash
cp .env.example .env
# 编辑 .env 文件，设置必要的配置项
```

2. 运行比赛：
```bash
python run_contest.py
```

## 目录结构

```
OpenManus/                      # OpenManus 框架目录
├── app/                        # 框架核心
│   ├── llm.py                 # LLM 基类
│   ├── config.py              # 配置管理
│   └── schema.py              # 数据模型
├── config/                     # 配置目录
│   └── config.toml            # 框架配置文件
└── projects/                   # 项目目录
    └── VerseMind/             # 诗歌比赛项目
        ├── poetry_llm.py      # 诗歌 LLM 实现
        ├── poetry_contest.py  # 比赛逻辑
        ├── run_contest.py     # 运行入口
        ├── prompts/          # 提示词模板
        │   ├── creation_prompt.txt    # 创作提示词
        │   ├── scoring_prompt.txt     # 评分提示词
        │   └── reference_content.txt  # 参考内容
        └── results/          # 结果输出目录（自动创建）
            ├── human_poem_example.txt # 人类诗歌示例（可选）
            ├── anonymous_poems_*.txt  # 匿名诗歌文件
            ├── scores_*.txt          # 评分报告
            └── final_ranking_*.txt   # 最终排名
```

## 使用说明

1. **创建诗人**：
   - 在配置文件中设置诗人信息
   - 支持多个诗人同时参与

2. **设置主题**：
   - 在 `prompts/reference_content.txt` 中设置比赛主题
   - 可以包含参考内容和要求

3. **添加人类诗歌**：
   - 将人类创作的诗歌保存在 `results/human_poem_example.txt` 中
   - 系统会自动将人类诗歌纳入评价流程

4. **运行比赛**：
   - 执行 `python run_contest.py`
   - 系统会自动完成创作和评价

5. **查看结果**：
   - 所有诗歌：`results/anonymous_poems_*.txt`
   - 评分结果：`results/scores_*.txt`
   - 最终排名：`results/final_ranking_*.txt`

## 许可证

MIT License 