import asyncio
import sys
from pathlib import Path

# 获取 OpenManus 根目录
openmanus_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(openmanus_root))

from app.agent.manus import Manus
from app.logger import logger

# 测试用的提示词
TEST_PROMPT = "写一首关于人工智能的诗"

# 要测试的 LLM 配置列表
LLM_CONFIGS = [
    {"name": "DeepSeek R1 14B", "config": "llm.deepseek-r1"},
    {"name": "Llama 3.2 Vision", "config": "llm.llama3.2-vision"},
    {"name": "Gemma 3 4B", "config": "llm.gemma3"},
    {"name": "Phi 4", "config": "llm.phi4"},
    {"name": "Mistral", "config": "llm.mistral"},
    {"name": "LLaVA v1.6", "config": "llm.llava"},
    {"name": "Qwen 2.5 7B", "config": "llm.qwen2.5"},
    {"name": "DeepSeek API", "config": "llm.deepseek-api"}
]

async def test_llm(config_name: str, prompt: str):
    """测试单个 LLM 配置"""
    try:
        print(f"\n{'='*50}")
        print(f"开始测试 {config_name}")
        print(f"{'='*50}")
        
        # 创建 Manus 实例
        agent = Manus(llm_config=config_name)
        
        # 运行测试
        print(f"发送提示词: {prompt}")
        response = await agent.llm.ask([{"role": "user", "content": prompt}])
        
        print(f"\n{config_name} 响应:\n{'-'*30}\n{response}\n{'-'*30}")
        return True
    except Exception as e:
        print(f"{config_name} 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """主函数：测试所有 LLM 配置"""
    print("\n开始 LLM 测试...")
    print(f"测试提示词: {TEST_PROMPT}")
    
    results = []
    for llm_config in LLM_CONFIGS:
        success = await test_llm(llm_config["config"], TEST_PROMPT)
        results.append({
            "name": llm_config["name"],
            "success": success
        })
        
        # 等待一段时间再测试下一个
        await asyncio.sleep(2)
    
    # 打印测试总结
    print("\n测试总结:")
    print("="*50)
    for result in results:
        status = "成功" if result["success"] else "失败"
        print(f"{result['name']}: {status}")
    print("="*50)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.warning("\nOperation interrupted by user.")
        sys.exit(0) 