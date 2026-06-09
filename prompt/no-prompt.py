from langchain_core.messages import SystemMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek

# 1. 创建模型
model = ChatDeepSeek(model_name="deepseek-chat")
#model = ChatOpenAI(model_name="gpt-4-turbo")
#model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# 2. 准备prompt
msg = [
    HumanMessage(content="我的账号突然登不上去了，急急急"),
]

result = model.invoke(msg)
print(result.content)
