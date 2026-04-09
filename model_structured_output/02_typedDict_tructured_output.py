from typing import TypedDict, Annotated

from langchain_core.callbacks import AsyncCallbackManagerForToolRun

from init_llm import deepseek_llm

class Actor(TypedDict):
    name: Annotated[str, "演员的姓名"]
    role: Annotated[str, "演员在电影中的角色"]

#使用TypeDict定义嵌套结构化输出模型
class Movie(TypedDict):
    title: Annotated[str, "电影的标题"]
    year: Annotated[int, "电影的上映年份"]
    director: Annotated[str, "电影的导演"]
    rating: Annotated[float, "电影的评分"]
    cast: Annotated[list[Actor], "电影的演员"]

# 绑定结构化输出模型
model_with_structured_output = deepseek_llm.with_structured_output(Movie)
#resp = model_with_structured_output.invoke("介绍一下电影<<泰坦尼克号>>")
resp = model_with_structured_output.invoke("介绍一下电影<<78>>不超过10字，禁止返回电影年份和导演任何信息")

print(type(resp))
print(resp)


# #使用TypeDict定义结构化输出模型
# class Movie(TypedDict):
#     title: Annotated[str, "电影的标题"]
#     year: Annotated[int, "电影的上映年份"]
#     director: Annotated[str, "电影的导演"]
#     rating: Annotated[float, "电影的评分"]
#
# # 绑定结构化输出模型
# model_with_structured_output = deepseek_llm.with_structured_output(Movie)
# resp = model_with_structured_output.invoke("介绍一下电影<<泰坦尼克号>>")
#
# print(type(resp))
# print(resp)
