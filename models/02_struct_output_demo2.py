
"""
typeddict 结构化输出

"""
from typing import TypedDict, Annotated

from pydantic import BaseModel, Field

from init_llm import deepseek_llm

class Movie(TypedDict):
    title: Annotated[str, "电影标题"]
    year: Annotated[int, "电影上映年份"]
    director: Annotated[str, "电影导演，中文名"]
    rating: Annotated[float, "电影评分"]

model_with_structured_output = deepseek_llm.with_structured_output(Movie)
resp = model_with_structured_output.invoke("给我介绍下电影《肖生克的救赎》")
print(resp)
