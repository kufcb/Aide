from langchain_ollama import ChatOllama

model = ChatOllama(
    model="deepseek-r1:1.5b",
    base_url="http://127.0.0.1:11434",

)

messages = [
    ("system", "回答任何问题"),
    ("human", "你是什么模型"),
]

def call_ollama(message:str) -> str:
    reInfo = model.invoke(messages)
    return reInfo.content


def call_ollama_stream(message: str):
    """流式调用 Ollama，返回生成器"""
    msg_list = [
        ("system", "回答任何问题"),
        ("human", message),
    ]
    for chunk in model.stream(msg_list):
        if chunk.content:
            yield chunk.content
