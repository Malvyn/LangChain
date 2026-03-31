import os
from typing import overload

from dotenv import load_dotenv

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

DASHSCOPE_API_KEY = os.getenv("Qianw_API_KEY")
DASHSCOPE_API_URL = os.getenv("Qianw_BASE_URL")


