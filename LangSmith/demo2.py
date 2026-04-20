from fastapi import FastAPI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAI, ChatOpenAI
import os

from langserve import add_routes

# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'


os.environ['LANGSMITH_PROJECT'] = 'Langchain-chat'

# 聊天机器人
# 1. 创建模型
model = ChatDeepSeek(model_name="deepseek-chat")
# model = ChatOpenAI(model_name="gpt-4-turbo")
# model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# 定义提示模板
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个非常乐于助人的助手.你必须严格只用{language}回答"),
    MessagesPlaceholder(variable_name="my_msg")
])

# 4. 得到链
chain = prompt_template | model

# 5. 保存聊天历史记录
store = {}  # 所有用户的聊天记录都保存在store里面。 key: session_id, value: 消息历史记录对象


# 此函数预期将接受一个session_id并返回一个消息历史记录对象
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store.get(session_id)


do_message = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key='my_msg'  # 每次聊天时候发送msg的key
)

config = {'configurable': {'session_id': 'zs123'}} #给当前会话定义一个session_id, 用于保存聊天历史记录

# 第一轮
resp = do_message.invoke(
    {"my_msg": [HumanMessage(content="你好啊,我是张三")],
     'language': '中文'
     },
    config=config
)

print(resp.content)

# 第二轮
resp2 = do_message.invoke(
    {"my_msg": [HumanMessage(content="请问我的名字是什么？")],
     'language': '中文'
     },
    config=config
)
print(resp2.content)

# 第三轮：返回的数据是流式的
config  = {'configurable': {'session_id': 'ls2323'}} #给当前会话定义一个session_id, 用于保存聊天历史记录
for resp in do_message.stream({"my_msg": [HumanMessage(content="讲一个笑话")],'language': 'English'},config=config):
    # 每一次resp都是一个token
    print(resp.content, end='-')
