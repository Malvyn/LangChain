"模型调用工具"
from langchain_core.messages import HumanMessage

from init_llm import deepseek_llm
from langchain_core.tools import tool


# 1. 创建工具
@tool
def get_weather(local: str) -> str:
    """获取天气信息"""
    return f"{local}的天气非常晴朗!"

# 2. 给模型绑定工具
model_with_tools = deepseek_llm.bind_tools(
    tools=[get_weather]
)

# 3. 准备messages
messages = []
humanMessage = HumanMessage(content="北京的天气是什么")
messages.append(humanMessage)


# 3. 模型不会真正执行调用工具，只是调用工具
response = model_with_tools.invoke(messages)
messages.append(response)

# 获取工具调用结果
if response.tool_calls:
    for tool_call in response.tool_calls:
        # 打印工具调用结果
        print(tool_call)
        if tool_call["name"] == "get_weather":
            # 手动调用工具
            tool_result = get_weather.invoke(tool_call)
            messages.append(tool_result)

# 6. final message
#print("messages:", messages)
final_response = model_with_tools.invoke(messages)
print("final_response:", final_response)

