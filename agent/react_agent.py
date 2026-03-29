import json
from typing import Annotated, Sequence, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from chat.zhipu_chat import chat as llm
from logs.logging_server import logger
from memory import get_memory_service
from tools.can_tools import (
    write_to_file,
    duckduckgo_search,
    read_file,
    run_terminal_command,
)


class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_id: str
    session_id: str
    agent_id: str
    current_user_input: str
    memory_context: str
    user_memory_written: bool


tools = [write_to_file, duckduckgo_search, read_file, run_terminal_command]

tools_by_name = {tool.name: tool for tool in tools}

model = llm.bind_tools(tools)
memory_service = get_memory_service()


def _latest_user_text(messages: Sequence[BaseMessage]) -> str:
    for message in reversed(messages):
        msg_type = getattr(message, "type", "")
        if msg_type == "human":
            return getattr(message, "content", "") or ""
    return ""


def _coerce_content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    chunks.append(str(text))
        return "\n".join(chunks).strip()
    if content is None:
        return ""
    return str(content)


def call_model(state: AgentState):
    user_id = state.get("user_id", "default_user")
    session_id = state.get("session_id", "default_session")
    current_user_input = state.get("current_user_input") or _latest_user_text(
        state["messages"]
    )
    user_memory_written = state.get("user_memory_written", False)
    if not user_memory_written and current_user_input.strip():
        memory_service.write_user_input(
            user_input=current_user_input,
            user_id=user_id,
            session_id=session_id,
        )
        user_memory_written = True

    memory_context = state.get("memory_context")
    if memory_context is None:
        memory_context = memory_service.build_prompt_context(
            user_input=current_user_input,
            user_id=user_id,
            session_id=session_id,
        )

    system_content = "你是一个 AI 助手。如果需要，你可以使用工具来获取信息来构建你的答案。"
    if memory_context:
        system_content += f"\n\n{memory_context}"

    system_prompt = SystemMessage(content=system_content)
    response = model.invoke([system_prompt] + list(state["messages"]))

    updates = {
        "messages": [response],
        "current_user_input": current_user_input,
        "user_memory_written": user_memory_written,
    }
    if state.get("memory_context") is None:
        updates["memory_context"] = memory_context

    tool_calls = getattr(response, "tool_calls", None)
    content = _coerce_content_to_text(getattr(response, "content", ""))
    if not tool_calls and content.strip():
        memory_service.write_turn(
            user_input=current_user_input,
            assistant_output=content,
            user_id=user_id,
            session_id=session_id,
        )

    return updates


def tool_node(state: AgentState):
    outputs = []

    last_message = state["messages"][-1]

    # 检查是否有工具调用请求
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        # 遍历每个工具调用
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # 根据工具名称找到对应的工具函数并执行
            if tool_name in tools_by_name:
                result = tools_by_name[tool_name].invoke(tool_args)
            else:
                result = f"工具 '{tool_name}' 不存在"

            # 将结果包装为 ToolMessage
            # ToolMessage 会被 LLM 读取，作为 Observation（观察结果）
            outputs.append(
                ToolMessage(
                    content=json.dumps(
                        result, ensure_ascii=False
                    ),  # 工具结果转为 JSON 字符串
                    name=tool_name,
                    tool_call_id=tool_call.get("id"),  # 关联对应的 tool_call
                )
            )

    # 返回工具执行结果，会被追加到消息历史
    return {"messages": outputs}


def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]

    # 检查是否有工具调用请求
    if not (hasattr(last_message, "tool_calls") and last_message.tool_calls):
        # 没有工具调用，任务完成
        return "end"
    else:
        # 有工具调用，需要执行工具后继续
        return "continue"


# 创建状态图
graph_builder = StateGraph(AgentState)

# 添加节点
graph_builder.add_node("call_model", call_model)  # LLM 推理节点
graph_builder.add_node("tool_node", tool_node)  # 工具执行节点

# 设置入口点：从 START 进入 call_model
graph_builder.set_entry_point("call_model")

# 添加普通边：tool_node 执行完后，回到 call_model 继续推理
graph_builder.add_edge("tool_node", "call_model")

# 添加条件边：从 call_model 出发，根据 should_continue 的结果决定去向
graph_builder.add_conditional_edges(
    "call_model",
    should_continue,
    {
        "continue": "tool_node",  # 继续循环：去执行工具
        "end": END,  # 结束循环：任务完成
    },
)

# 编译图，生成可执行的工作流
graph = graph_builder.compile()

async def agent_main_stream(
    user_input: str,
    user_id: str = "default_user",
    session_id: str = "default_session",
    agent_id: str = "aide-react-agent",
):
    inputs = {
        "messages": [("user", user_input)],
        "user_id": user_id,
        "session_id": session_id,
        "agent_id": agent_id,
        "current_user_input": user_input,
    }

    thought_started = False
    answer_started = False

    for event in graph.stream(inputs, stream_mode="values"):
        messages = event.get("messages", [])
        if messages:
            last_message = messages[-1]
            logger.info("返回信息:")
            logger.info(last_message)
            msg_type = getattr(last_message, "type", "unknown")
            content = getattr(last_message, "content", "")

            if msg_type == "ai":
                # AI 回复
                tool_calls = getattr(last_message, "tool_calls", None)
                if tool_calls:
                    # 工具调用请求 - 开始思考过程
                    if not thought_started:
                        thought_started = True
                        tool_names = [tc.get("name", "unknown") for tc in tool_calls]
                        yield f"[THOUGHT_START]|||正在思考... 调用工具: {tool_names}"
                    else:
                        # 继续添加思考内容
                        tool_names = [tc.get("name", "unknown") for tc in tool_calls]
                        yield f"\n调用工具: {tool_names}"

                elif content:
                    # 最终答案 - 关闭思考区域，开始答案区域
                    if thought_started:
                        thought_started = False
                        yield "|||[THOUGHT_END]\n"
                    if not answer_started:
                        answer_started = True
                        yield f"[ANSWER_START]|||{content}"
                    else:
                        yield content

            elif msg_type == "tool":
                # 工具执行结果 - 添加到思考过程
                if thought_started:
                    try:
                        result = json.loads(content)
                        # 简化输出，只显示摘要
                        if isinstance(result, list) and len(result) > 0:
                            summary = f"获取到 {len(result)} 条结果"
                        else:
                            summary = (
                                str(result)[:100] + "..."
                                if len(str(result)) > 100
                                else str(result)
                            )
                        yield f"\n工具返回: {summary}"
                    except:
                        summary = (
                            content[:100] + "..." if len(content) > 100 else content
                        )
                        yield f"\n工具返回: {summary}"

    # 确保关闭所有区域
    if thought_started:
        yield "|||[THOUGHT_END]\n"
    if answer_started:
        yield "|||[ANSWER_END]\n"
