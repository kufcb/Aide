"""
使用 LangChain 官方 create_react_agent 实现 ReAct 模式
"""
import json
import asyncio
from typing import AsyncGenerator

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from chat.zhipu_chat import chat as zhipu_chat_model
from tools.file_tool import write_to_file, read_file
from tools.terminal_tool import run_terminal_command


# ReAct 官方 Prompt 模板
REACT_TEMPLATE = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Tool input rules:
- read_file and run_terminal_command expect a plain string as Action Input.
- write_to_file expects a JSON object string like {{"file_path": "example.txt", "content": "hello"}}.

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""


def create_react_agent_executor():
    """创建 ReAct Agent Executor"""

    def write_to_file_tool(action_input: str) -> str:
        """兼容 ReAct 单字符串输入的写文件工具包装器。"""
        try:
            payload = json.loads(action_input)
        except json.JSONDecodeError as exc:
            return f"写入失败：Action Input 必须是 JSON，错误：{exc}"

        file_path = payload.get("file_path")
        content = payload.get("content")
        if not file_path or content is None:
            return "写入失败：JSON 中必须包含 file_path 和 content 字段"

        return write_to_file(file_path=file_path, content=content)

    # 1. 包装工具为 LangChain Tool 对象
    tools = [
        Tool(
            name="read_file",
            func=read_file,
            description="读取指定文件的内容。输入参数：file_path (str) 文件路径"
        ),
        Tool(
            name="write_to_file", 
            func=write_to_file_tool,
            description='将内容写入指定文件。Action Input 必须是 JSON 字符串，例如 {"file_path": "test.txt", "content": "Hello World"}'
        ),
        Tool(
            name="run_terminal_command",
            func=run_terminal_command,
            description="执行终端命令。输入参数：command (str) 要执行的命令"
        ),
    ]
    
    # 2. 创建 Prompt
    prompt = PromptTemplate.from_template(REACT_TEMPLATE)
    
    # 3. 创建 ReAct Agent
    agent = create_react_agent(
        llm=zhipu_chat_model,
        tools=tools,
        prompt=prompt
    )
    
    # 4. 创建 AgentExecutor（控制执行循环）
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # 打印详细执行过程
        handle_parsing_errors=True,  # 自动处理解析错误
        max_iterations=10,  # 最大迭代次数，防止死循环
    )
    
    return agent_executor


def run_agent(user_input: str):
    """运行 Agent 并返回结果"""
    agent_executor = create_react_agent_executor()
    result = agent_executor.invoke({"input": user_input})
    return result["output"]



def agent_main(user_input: str):
    response = run_agent(user_input)
    print(f"\n最终答案：{response}")
    return response


async def run_agent_stream(user_input: str) -> AsyncGenerator[str, None]:
    """
    流式运行 Agent，产生 SSE 格式的数据流
    
    生成的事件格式：
    - event: thought\ndata: <思考内容>\n\n
    - event: action\ndata: <动作内容>\n\n
    - event: observation\ndata: <观察结果>\n\n
    - event: final\ndata: <最终答案>\n\n
    """
    agent_executor = create_react_agent_executor()
    
    # 使用 astream 方法获取流式输出
    async for event in agent_executor.astream_events({"input": user_input}, version="v1"):
        event_type = event["event"]
        
        if event_type == "on_chain_start":
            if event.get("name") == "Agent":
                yield f"event: start\ndata: Agent started\n\n"
        
        elif event_type == "on_chain_end":
            if event.get("name") == "Agent":
                output = event["data"].get("output", "")
                if output:
                    yield f"event: final\ndata: {json.dumps({'output': output}, ensure_ascii=False)}\n\n"
        
        elif event_type == "on_chat_model_stream":
            # 获取模型生成的 token
            chunk = event["data"].get("chunk", {})
            content = getattr(chunk, 'content', '') if hasattr(chunk, 'content') else str(chunk)
            if content:
                yield f"event: token\ndata: {json.dumps({'token': content}, ensure_ascii=False)}\n\n"
        
        elif event_type == "on_tool_start":
            tool_name = event.get("name", "")
            tool_input = event["data"].get("input", "")
            yield f"event: action\ndata: {json.dumps({'tool': tool_name, 'input': tool_input}, ensure_ascii=False)}\n\n"
        
        elif event_type == "on_tool_end":
            tool_name = event.get("name", "")
            output = event["data"].get("output", "")
            yield f"event: observation\ndata: {json.dumps({'tool': tool_name, 'output': output}, ensure_ascii=False)}\n\n"
    
    yield f"event: done\ndata: Stream completed\n\n"


def agent_main_stream(user_input: str):
    """
    同步包装器，返回异步生成器
    用于 FastAPI StreamingResponse
    """
    return run_agent_stream(user_input)
