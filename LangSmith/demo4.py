from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_chroma import Chroma
from langchain_tavily import TavilySearch
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda, RunnablePassthrough
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAI, ChatOpenAI, OpenAIEmbeddings
import os

from langchain.agents import create_agent
from langserve import add_routes

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'

os.environ['LANGSMITH_API_KEY'] = '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'
os.environ['LANGSMITH_TRACING'] = 'true'
os.environ['LANGSMITH_PROJECT'] = 'Langchain-proxy'

load_dotenv()  # 默认查找当前目录的 .env 文件
# 聊天机器人
# 1. 创建模型
model = ChatDeepSeek(model_name="deepseek-chat")

#没有任何代理的情况下
#result = model.invoke([HumanMessage(content="北京天气怎么样？")])
#print(result.content)

#langchain 内置了一个工具，可以轻松地使用Tavily 搜索引擎作为工具
search = TavilySearch(max_results =2, api_key=os.getenv('TAVILY_API_KEY')) # max_results :只返回两个结果
#print(search.invoke("北京天气怎么样？"))

# 2. 让模型绑定工具
tools = [search]
#model_with_tools = model.bind_tools([tools])
# 模型可以自动推理：是否需要调用工具去完成用户的答案
# resp = model_with_tools.invoke([HumanMessage(content="中国的首都是哪个城市？")])
#
# print(f'Model_Result_Content:{resp.content}')
# print(f'Model_Result_Content:{resp.tool_calls}')
#
# resp2 = model_with_tools.invoke([HumanMessage(content="北京天气怎么样？")])
# print(f'Model_Result_Content:{resp2.content}')
# print(f'Model_Result_Content:{resp2.tool_calls}')

#创建代理
agent_executor = create_agent(model, tools)
resp = agent_executor.invoke({"messages": [HumanMessage(content="中国的首都是哪个城市？")]})
print(resp["messages"])

resp2 = agent_executor.invoke({"messages": [HumanMessage(content="北京天气怎么样？")]})
print(resp2["messages"])
