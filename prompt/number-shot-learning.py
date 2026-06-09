import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

from my_llm import deepseek_llm, google_llm

model = google_llm

prompt = """
这组数字中的奇数加起来是一个偶数： 4、8、9、15、12、2、1。
A: 答案是False。

这组数字中的奇数加起来是一个偶数： 1、10、19、4、8、12、24。
A: 答案是True。

这组数字中的奇数加起来是一个偶数： 16、11、14、4、8、13、24。
A: 答案是True。

这组数字中的奇数加起来是一个偶数： 17、9、10、12、13、4、2。
A: 答案是False。

这组数字中的奇数加起来是一个偶数： 15、32、5、13、82、7、1。
A:
"""

resp = model.invoke(prompt)
print(resp.content)
