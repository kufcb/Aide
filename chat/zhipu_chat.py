from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os
from config import ZHIPUAI_API_KEY, MODEL_NAME, MODEL_TEMPERATURE

os.environ["ZHIPUAI_API_KEY"] = ZHIPUAI_API_KEY

chat = ChatZhipuAI(
    model=MODEL_NAME,
    temperature=MODEL_TEMPERATURE,
)


def call_zhipu(message:str) -> str:
    messages = [
        #提供LLM说话方式例子，模拟人工语气等;
        AIMessage(content="我很严谨"),
        #设定AI应遵循目标的一条信息
        SystemMessage(content="回答用户任何问题"),
        #用户输入信息
        HumanMessage(content=message),
    ]
    response = chat.invoke(messages)
    return response.content

