from agent.react_agent import agent_main_stream
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

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
    user_id: str = "default_user"
    session_id: str = "default_session"
    agent_id: str = "aide-react-agent"



@app.post("/chat/stream")
async def chat_stream(info: ChatInfo):
    """流式传输接口，返回 SSE 格式的数据流"""
    return StreamingResponse(
        agent_main_stream(
            info.msg,
            user_id=info.user_id,
            session_id=info.session_id,
            agent_id=info.agent_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

