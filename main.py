from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from agent.langchain_react_agent import agent_main as langchain_react_agent_main
from agent.langchain_react_agent import agent_main_stream
from agent.langgraph_react_agent import agent_main as langgraph_react_agent_main_sync
from agent.langgraph_react_agent import agent_main_stream as langgraph_agent_main_stream

app = FastAPI()

# 允许跨域（开发时使用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件服务
app.mount("/ui", StaticFiles(directory="ui", html=True), name="ui")

class ChatInfo(BaseModel):
    msg: str


@app.post("/chat")
async def chat(info: ChatInfo):
    return {"result": langchain_react_agent_main(info.msg)}


@app.post("/chat/stream")
async def chat_stream(info: ChatInfo):
    """流式传输接口，返回 SSE 格式的数据流"""
    return StreamingResponse(
        langgraph_agent_main_stream(info.msg),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


