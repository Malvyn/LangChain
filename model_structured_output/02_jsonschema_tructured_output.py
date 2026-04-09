from init_llm import deepseek_llm
import json

# 1. JSON Schema - 添加 title 字段
# json_schema = {
#     "title": "MovieInfo",  # 添加这一行，作为函数名
#     "type": "object",
#     "properties": {
#         "title": {"type": "string", "description": "电影标题"},
#         "year": {"type": "integer", "description": "上映年份"},
#         "director": {"type": "string", "description": "导演姓名"},
#         "rating": {"type": "number", "description": "评分"},
#         "cast": {
#             "type": "array",
#             "description": "电影演员列表",
#             "items": {
#                 "type": "object",
#                 "properties": {
#                     "name": {"type": "string"},
#                     "role": {"type": "string"}
#                 },
#                 "required": ["name", "role"]
#             }
#         }
#     },
#     "required": ["title", "year", "director", "rating", "cast"]
# }

json_schema = {
    "title": "MovieInfo",  # 添加这一行，作为函数名
    "description": "电影信息",
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "电影标题"},
        "year": {"type": "integer", "description": "上映年份"},
        "director": {"type": "string", "description": "导演姓名"},
        "rating": {"type": "number", "description": "评分"},
        "cast": {
            "type": "array",
            "description": "电影演员列表",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"}
                },
                "required": ["name", "role"],
            },
            "description": "电影演员列表"
        },
    },
    "required": ["title", "year", "director", "rating", "cast"]
}

# 2. Prompt - 提供具体的电影信息
prompt_content = "请介绍电影《盗梦空间》"

# 模型绑定结构化输出 jsonSchema
model_with_schema = deepseek_llm.with_structured_output(json_schema)

resp = model_with_schema.invoke(prompt_content)

print(resp)
