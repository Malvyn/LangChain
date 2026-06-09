import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from my_llm import deepseek_llm, google_llm

# prompt = f'''
# 根据下面的上下文回答问题，保持答案简短且准确，如果不确定答案，请回答"不确定答案"
#
# Teplizumab起源于一个位于新泽西的药品公司，名为Ortho Pharmaceuticals.\
# 在哪里，科学家们生成了一种早期版本的抗体，被称为OKT3。最初这种分子是从小白鼠中提取的。\
# 能欧结合到T细胞的表面，并限制它们的细胞杀伤力。在1986年，它被批准用于帮助预防肾脏移植后的\
# 器官排斥，成为首个被一些用于人类的治疗性抗体
#
#
# 问题： OKT3最初是从什么来源提取的？
#
# '''

# 定义结构，定义变量

# instruction = '''
# 根据下面的上下文回答问题，保持答案简短且准确，如果不确定答案，请回答"不确定答案"
# '''
# context = """
# 根据下面的上下文回答问题，保持答案简短且准确，如果不确定答案，请回答"不确定答案"
#
# Teplizumab起源于一个位于新泽西的药品公司，名为Ortho Pharmaceuticals.\
# 在哪里，科学家们生成了一种早期版本的抗体，被称为OKT3。最初这种分子是从小白鼠中提取的。\
# 能欧结合到T细胞的表面，并限制它们的细胞杀伤力。在1986年，它被批准用于帮助预防肾脏移植后的\
# 器官排斥，成为首个被一些用于人类的治疗性抗体
# """
#
# query = """
# OKT4最初是从什么来源提取的？
# """

# 添加格式
instruction = """
根据下面的上下文回答问题，保持答案简短且准确，如果不确定答案，请回答"不确定答案"   

以json格式输出：
{"[具体问题]": "[答案]"}, 
"""

examples ="""
{"你是谁？": "我的大模型"}, 
"""

context ="""
Teplizumab起源于一个位于新泽西的药品公司，名为Ortho Pharmaceuticals.\
在哪里，科学家们生成了一种早期版本的抗体，被称为OKT3。最初这种分子是从小白鼠中提取的。\
能欧结合到T细胞的表面，并限制它们的细胞杀伤力。在1986年，它被批准用于帮助预防肾脏移植后的\
器官排斥，成为首个被一些用于人类的治疗性抗体
"""

query = """
OKT3最初是从什么来源提取的？
"""

prompt = f"{instruction}\n{examples}\n{query}\n{context}\n{query}"

model = google_llm
resp = model.invoke(prompt)
print(resp.content)
