"""
项目配置文件
存放 API Key 等敏感信息
"""
import os
from dotenv import load_dotenv

load_dotenv()

# 智谱 AI API Key
# 优先从环境变量读取，如果没有则使用默认值
ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY")
# GLM-4.5-Air  GLM-4.7-Flash
# 模型配置
MODEL_NAME = os.getenv("MODEL_NAME", "GLM-4.7-Flash")
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.5"))
