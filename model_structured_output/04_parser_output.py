"pydantic 模型返回结构化数据"
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

deepseek_reasoner_llm = init_chat_model(
    model="deepseek-reasoner",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

class Movie(BaseModel):
    title: str = Field(description="电影标题")
    release_date: str = Field(description="上映日期")

# 2. 设计提示词
prompt = ChatPromptTemplate.from_template("""
回答用户的问题
问题：{question}
你必须始终输出一个包含title(电影标题)和release_date(上映日期)的JSON对象
""")

#3.创建链
chain = prompt | deepseek_reasoner_llm | JsonOutputParser(pydantic_object=Movie)

resp = chain.invoke({"question": "给我介绍下电影《肖生克的救赎》"})
print(resp)
