from fastapi import FastAPI
from langchain_chroma import Chroma
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

from langserve import add_routes

from env_utils import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL

# os.environ['http_proxy'] = 'http://127.0.0.1:7890'
# os.environ['https_proxy'] = 'http://127.0.0.1:7890'


os.environ['LANGSMITH_PROJECT'] = 'Langchain-vectorstore'

# 聊天机器人
# 1. 创建模型
model = ChatDeepSeek(model_name="deepseek-chat")

#准备测试数据，假设我们提供的文档数据如下：
documents = [
    Document(
        page_content="狗是伟大的伴侣，一起忠诚和友好而闻名。",
        metadata={"source": "哺乳动物宠物文档"}
    ),
    Document(
        page_content="猫是独立的宠物，他们自己的生活和行为。",
        metadata={"source": "哺乳动物宠物文档"}
    ),
    Document(
        page_content="金鱼是初学者的流行宠物，需要相对简单的护理。",
        metadata={"source": "鱼类动物宠物文档"}
    ),
    Document(
        page_content="鹦鹉是聪明的鸟类，能够模仿人类的语言。",
        metadata={"source": "鸟类动物宠物文档"}
    ),
    Document(
        page_content="兔子是社交动物，需要足够的空间跳跃。",
        metadata={"source": "哺乳动物宠物文档"}
    ),
]

# 2. 示例化向量数空间

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # 或使用 "deepseek-embedding"
)


vector_store = Chroma.from_documents(documents, embedding=embeddings)

# 相似度的查询： 返回相似的分数，分数越小，相似度越高，否则越低
#print(vector_store.similarity_search_with_score("咖啡猫"))

# 检索器: bind k=1 返回相似度最高的一个
retriever = RunnableLambda(vector_store.similarity_search_with_score).bind(k=1)
#print(retriever.batch(['咖啡猫','鲨鱼']))

# 提示模板
message ="""
使用提供的上下文仅回答这个问题
{question}
上下文：
{context}
"""

prompt_temp = ChatPromptTemplate.from_messages([("human", message)])

#RunnablePassthrough允许我们将用户的问题之后再传递给prompt和model
chain = {'question': RunnablePassthrough(),'context': retriever} | prompt_temp | model
resp = chain.invoke('请介绍一下猫？')
print(resp.content)



