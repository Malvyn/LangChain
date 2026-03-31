
"""
pydantic 结构化输出

"""
from pydantic import BaseModel, Field

from init_llm import deepseek_llm

class Movie(BaseModel):
    title: str = Field(description="电影标题")
    director: str = Field(description="导演")
    actor: str = Field(description="演员")
    genre: str = Field(description="电影类型")
    release_date: str = Field(description="上映日期")
    duration: str = Field(description="电影时长")
    description: str = Field(description="电影描述")

model_with_structured_output = deepseek_llm.with_structured_output(Movie)
resp = model_with_structured_output.invoke("给我介绍下电影《肖生克的救赎》")
print(resp)
