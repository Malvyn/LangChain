from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from typing import Literal
from pydantic import BaseModel, Field

model = ChatDeepSeek(model_name="deepseek-chat")


# class Classification(BaseModel):
#    """
#    定义一个pydantic模型，未来需要根据该类型，完成文本的分类
#
#    """
#    # 文本的情感倾向，预期为字符串类型
#    sentiment: str = Field(description="文本的情感倾向")
#
#    #文本的攻击性，预期为1到10的整数
#    aggressiveness: int = Field(
#       description="描述文本的攻击性，数字越大表示越攻击性")
#
#    # 文本使用的语言，预期为字符串类型
#    language: str = Field(description="文本使用的语言")

class Classification(BaseModel):
   """
   定义一个pydantic模型，未来需要根据该类型，完成文本的分类

   """
   # 文本的情感倾向，预期为字符串类型
   sentiment: Literal["happy", "neutral", "sad"] = Field(description="文本的情感倾向")

   # 文本的攻击性，预期为1到10的整数
   aggressiveness: Literal[1, 2, 3, 4, 5] = Field(description="描述文本的攻击性，数字越大表示越攻击性")

   # 文本使用的语言，预期为字符串类型
   language: Literal["spanish", "english", "french", "中文", "italian"] = Field(description="文本使用的语言")

# 创建一个用于提取信息的提示模板

tagging_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个文本分析专家。从以下段落中提取所需的信息，只提取Classification类中定义的字段。"),
    ("human", "段落：{input}")
])

chain = tagging_prompt | model.with_structured_output(Classification)

input_text = "中国人民大学教授：师德败坏，做出的事情实在让人生气！"
result: Classification = chain.invoke({"input": input_text})
print(result)
