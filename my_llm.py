import os

from langchain_community.embeddings import DashScopeEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DASHSCOPE_API_KEY, GOOGLE_API_KEY

# 两种方式创建大模型 对象
#1. 直接创建大模型对象
deepseek_llm = ChatDeepSeek(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    model="deepseek-chat",
)

# response = deepseek_llm.invoke("介绍一下你自己")
# print(response.content)


# embeddings = DashScopeEmbeddings(
#     model="text-embedding-v3",
#     dashscope_api_key=DASHSCOPE_API_KEY,
# )
#

google_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
)
# 同步调用
# response = google_llm.invoke("请介绍一下你自己")
# print(response.content)


OpenAI_llm = ChatOpenAI(
    model="gpt-5.2",
    temperature=0.7,
)
response = OpenAI_llm.invoke("请介绍一下你自己")
print(response.content)


