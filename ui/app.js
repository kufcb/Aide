const API_BASE_URL = 'http://localhost:8000';
const chatContainer = document.getElementById('chatContainer');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');

let isLoading = false;

// 回车发送
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !isLoading) {
        sendMessage();
    }
});

// 移除欢迎消息
function removeWelcomeMessage() {
    const welcomeMsg = document.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
}

function addMessage(content, isUser = false, isStreaming = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isUser ? 'user' : 'assistant'} ${isStreaming ? 'streaming' : ''}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.textContent = content;
    
    messageDiv.appendChild(contentDiv);
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return { messageDiv, contentDiv };
}

function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error';
    errorDiv.textContent = message;
    chatContainer.appendChild(errorDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    setTimeout(() => errorDiv.remove(), 5000);
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message || isLoading) return;

    isLoading = true;
    sendBtn.disabled = true;
    messageInput.value = '';

    // 移除欢迎消息
    removeWelcomeMessage();

    // 添加用户消息
    addMessage(message, true);

    try {
        await sendStreamRequest(message);
    } catch (error) {
        showError(error.message);
    } finally {
        isLoading = false;
        sendBtn.disabled = false;
        messageInput.focus();
    }
}

// 流式模式请求 - 使用 LangGraph Agent（支持思考过程显示）
async function sendStreamRequest(message) {
    const { messageDiv, contentDiv } = addMessage('', false, true);
    
    // 创建思考区域和答案区域
    const thoughtDiv = document.createElement('div');
    thoughtDiv.className = 'thought-process';
    thoughtDiv.style.display = 'none';
    
    const answerDiv = document.createElement('div');
    answerDiv.className = 'answer-content';
    
    contentDiv.innerHTML = '';
    contentDiv.appendChild(thoughtDiv);
    contentDiv.appendChild(answerDiv);

    const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ msg: message })
    });

    if (!response.ok) {
        throw new Error(`请求失败: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    let buffer = '';
    let currentThought = '';
    let currentAnswer = '';
    let inThought = false;
    let inAnswer = false;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;
        
        // 解析标记
        while (buffer.length > 0) {
            if (!inThought && !inAnswer) {
                // 查找开始标记
                const thoughtStart = buffer.indexOf('[THOUGHT_START]|||');
                const answerStart = buffer.indexOf('[ANSWER_START]|||');
                
                if (thoughtStart !== -1) {
                    inThought = true;
                    buffer = buffer.substring(thoughtStart + '[THOUGHT_START]|||'.length);
                    thoughtDiv.style.display = 'block';
                } else if (answerStart !== -1) {
                    inAnswer = true;
                    buffer = buffer.substring(answerStart + '[ANSWER_START]|||'.length);
                } else {
                    // 没有完整标记，保留缓冲区
                    break;
                }
            } else if (inThought) {
                // 查找思考结束标记
                const thoughtEnd = buffer.indexOf('|||[THOUGHT_END]');
                if (thoughtEnd !== -1) {
                    currentThought += buffer.substring(0, thoughtEnd);
                    thoughtDiv.textContent = currentThought;
                    buffer = buffer.substring(thoughtEnd + '|||[THOUGHT_END]'.length);
                    inThought = false;
                } else {
                    // 还没有结束标记，显示当前内容
                    currentThought += buffer;
                    thoughtDiv.textContent = currentThought;
                    buffer = '';
                }
            } else if (inAnswer) {
                // 查找答案结束标记
                const answerEnd = buffer.indexOf('|||[ANSWER_END]');
                if (answerEnd !== -1) {
                    currentAnswer += buffer.substring(0, answerEnd);
                    answerDiv.textContent = currentAnswer;
                    buffer = buffer.substring(answerEnd + '|||[ANSWER_END]'.length);
                    inAnswer = false;
                } else {
                    // 还没有结束标记，显示当前内容
                    currentAnswer += buffer;
                    answerDiv.textContent = currentAnswer;
                    buffer = '';
                }
            }
        }
        
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    // 移除流式状态
    messageDiv.classList.remove('streaming');
}
