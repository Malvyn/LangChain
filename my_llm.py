import os

from langchain_deepseek import ChatDeepSeek
from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL


# 两种方式创建大模型 对象
#1. 直接创建大模型对象
deepseek_llm = ChatDeepSeek(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    model="deepseek-chat",
)

resp = deepseek_llm.invoke("你好")
print(type(resp))
print(resp)
print(resp.content)


