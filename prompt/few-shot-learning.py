import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from my_llm import deepseek_llm, google_llm

model = google_llm

# prompt ="""
# 将文本分类为中性、负面或正面。
# 文本：明天要放假了，又要带娃啊、好烦啊。
# 情感：
# """

# prompt = """
# "whatpu"是坦桑尼亚的一种小型毛蓉蓉的动物。一个使用whatpu这个词的句子的例子是：
# 我们在非洲旅行时看到了这些非常可爱的whatpus。
# "farduddle"是指快速跳上跳下。一个使用fraduddle这个词的句子的例子是：
# """

# prompt = """
# "调优"本身是一个词，但是呢有人喜欢这样造句： 周深的声调优于白鹿
# 请仿照例子，使用“造句”造句
# """

prompt = """
"调优"本身是一个词，但是呢有人喜欢故意拆开这个词的意思来造句： 周深的声调优于白鹿
请仿照例子，使用“开心”造句
"""

resp = model.invoke(prompt)
print(resp.content)
