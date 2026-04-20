from fastapi import FastAPI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAI, ChatOpenAI
import os

from langserve import add_routes

#os.environ['http_proxy'] = 'http://127.0.0.1:7890'
#os.environ['https_proxy'] = 'http://127.0.0.1:7890'



# 1. 创建模型
model = ChatDeepSeek(model_name="deepseek-chat")
#model = ChatOpenAI(model_name="gpt-4-turbo")
#model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# 2. 准备prompt
# msg = [
#     SystemMessage(content="请将以下的内容翻译成意大利语"),
#     HumanMessage(content="你好,请问你要去哪里"),
# ]

# result = model.invoke(msg)
# print(result)

# 3. 创建返回数据的解析器
parser = StrOutputParser()
# print(parser.invoke(result))

# 定义提示模板
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "请将下面的内容翻译成{language}"),
    ('user', "{text}"),
])

# 4. 得到链
#chain = model | parser
chain = prompt_template | model | parser

# 5. 直接使用chain来调用
# print(chain.invoke(msg))
print(chain.invoke({"language": "English", "text": "我下午还有一节课,不能去打球了。"}))

# 把我们的程序部署成服务
# 创建fastAPI服务

app = FastAPI(title = '我的langchain服务', version = '1.0', description = '这是一个基于langchain的翻译服务')

add_routes(app,
           chain,
           path="/translate")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)
