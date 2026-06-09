from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

# 1. 创建模型
model = ChatDeepSeek(model_name="deepseek-chat")
#model = ChatOpenAI(model_name="gpt-4-turbo")
#model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# 2. 准备prompt

# msg = [
#     SystemMessage(content="你是一个专业的客服工单处理员."),
#     HumanMessage(content="我的账号突然登不上去了，急急急"),
# ]

# msg = [
#     SystemMessage(content="你是一个专业的客服工单处理员."),
#     HumanMessage(content="{user_question}"),
# ]

msg = [
    ("system", "你是一个专业的客服工单处理员."),
    ('user', "{user_question}"),
]

# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", "你是一个专业的客服工单处理员."),
#     ('user', "{user_question}"),
# ])

prompt_template = ChatPromptTemplate.from_messages(msg)
result = model.invoke(prompt_template.format(user_question="我的账号突然登不上去了，急急急"))
print(result.content)
