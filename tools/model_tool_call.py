from langchain.tools import tool
from langchain_core.messages import HumanMessage

from my_llm import deepseek_llm


# LLM 使用工具返回结果的步骤
#1. 定义工具，绑定工具到大模型
#2. 与LLM进行对话，LLM返回调用大模型的请求，并不会主动调用工具
#3. 根据返回的结果手动处理，并将结果告知LLM
#4. LLM 最后生成回复

@tool
def get_weather(city: str) -> str:
    """Get the current weather in a given location."""
    return f"The current weather in {city}的天气是晴朗，温度25摄氏度"

#1.绑定工具到大模型
model_bind_tool = deepseek_llm.bind_tools([get_weather])

messages = []
human_message = HumanMessage(content="北京天气")

messages.append(human_message)

#2.LLM 返回调用的工具请求
resp = model_bind_tool.invoke(messages)
messages.append(resp)

for tool_call in resp.tool_calls:
    if tool_call["name"] == "get_weather":
       tool_result = get_weather.invoke(tool_call)
       messages.append(tool_result)

resp = model_bind_tool.invoke(messages)
print('message', messages)
print(resp)




