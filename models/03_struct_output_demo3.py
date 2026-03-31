
"""
jsonschema 结构化输出

"""
from typing import TypedDict, Annotated

from pydantic import BaseModel, Field
from pydantic.v1.schema import json_scheme

from init_llm import deepseek_llm

#创建jsonschema 结构化输出的模型
json_scheme= {
    "title": "movie",
    "description": "电影的详细信息，包括标题、上映年份、导演、评分",
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "year": {"type": "integer"},
        "director": {"type": "string"},
        "rating": {"type": "number"},
    },
    "required": ["title", "year", "director", "rating"],
}

model_with_structured_output = deepseek_llm.with_structured_output(json_scheme)
resp = model_with_structured_output.invoke("给我介绍下电影《肖生克的救赎》")
print(resp)
