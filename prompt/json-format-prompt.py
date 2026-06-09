from langchain_core.messages import SystemMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek

# 1. 创建模型
model = ChatDeepSeek(model_name="deepseek-chat")
#model = ChatOpenAI(model_name="gpt-4-turbo")
#model = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# 2. 准备prompt

prompt_template = """
你是一个专业的客服工单处理员。

任务：将用户的问题转化为【标准客服工单标题】和【紧急程度】。

限制条件：
- 紧急程度只能是：低 / 中 / 高
- 工单标题不超过 20 个字

输出格式（严格 JSON，不要有其他文字）：
{{"title": "xxx", "urgency": "低/中/高"}}

示例1：
用户：如何修改绑定手机号？
输出：{{"title": "修改绑定手机号", "urgency": "低"}}

示例2：
用户：无法支付，但卡里有钱
输出：{{"title": "支付失败", "urgency": "高"}}

示例3：
用户：APP 偶尔闪退
输出：{{"title": "APP闪退问题", "urgency": "中"}}

现在请处理下面这个用户输入，只输出 JSON：

用户问题：{user_question}
"""
msg = [SystemMessage(content=prompt_template.format(user_question="我的账号突然登不上去了，急急急"))]

result = model.invoke(msg)
print(result.content)
