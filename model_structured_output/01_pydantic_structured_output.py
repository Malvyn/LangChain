from pydantic import BaseModel, Field

from my_llm import deepseek_llm

class Actor(BaseModel):
    name: str = Field(description="演员的姓名")
    role: str = Field(description="演员的角色")


#返回一个嵌套对象
class Movie(BaseModel):
    title: str = Field(description="电影的标题")
    year: int = Field(description="电影的上映年份")
    director: str = Field(description="电影的导演")
    rating: float = Field(description="电影的评分")
    cast: list[Actor] = Field(description="电影的演员")

# 使用正确的方法名 with_structured_output
#model_with_structured_output = deepseek_llm.with_structured_output(Movie)

model_with_structured_output = deepseek_llm.with_structured_output(Movie, include_raw=True)

resp = model_with_structured_output.invoke("介绍一下电影<<泰坦尼克号>>")
print(resp)



# 定义一个Pydantic模型，用于结构化输出简单对象
# class Movie(BaseModel):
#     title: str = Field(description="电影的标题")
#     year: int = Field(description="电影的上映年份")
#     director: str = Field(description="电影的导演")
#     actor: str = Field(description="电影的演员")
#     genre: str = Field(description="电影的类型")
#     summary: str = Field(description="电影的摘要")
#
# # 使用正确的方法名 with_structured_output
# model_with_structured_output = deepseek_llm.with_structured_output(Movie)
# #resp = model_with_structured_output.invoke("介绍一下电影<<泰坦尼克号>>")
# resp = model_with_structured_output.invoke("介绍一下电影<<泰坦尼克号>>,禁止返回电影年份和导演")
# print(resp)