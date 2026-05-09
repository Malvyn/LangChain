import os

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek
from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DASHSCOPE_API_KEY

# 两种方式创建大模型 对象
#1. 直接创建大模型对象
deepseek_llm = ChatDeepSeek(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    model="deepseek-chat",
)

response = deepseek_llm.invoke("介绍一下你自己")
print(response.content)


embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=DASHSCOPE_API_KEY,
)

resp = deepseek_llm.invoke("你好")
print(type(resp))
print(resp)
print(resp.content)


