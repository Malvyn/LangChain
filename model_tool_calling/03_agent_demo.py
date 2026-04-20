from venv import create

from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage  # 1. 导入 ToolMessage

from init_llm import deepseek_llm
from langchain.agents import create_agent

@tool
def get_stock_price(company: str, timeframe: str = "today") -> str:
    """获取股票价格信息。"""
    mock_data = {
        "苹果公司": {"today": 185.20, "week": 183.50, "month": 182.00, "year": 175.00},
        "特斯拉公司": {"today": 240.50, "week": 238.00, "month": 235.00, "year": 220.00},
        "谷歌公司": {"today": 142.50, "week": 141.00, "month": 139.50, "year": 130.00},
    }
    if company in mock_data:
        price = mock_data[company].get(timeframe, "未找到该时间范围的价格数据")
        return f"{company} {timeframe}股票价格: {price}美元"
    else:
        return f"未找到 {company} 的股票数据"


@tool
def search_news(company: str) -> str:
    """搜索股票新闻。"""
    mock_news = {
        "苹果公司": ["最近发布了一款新的产品。", "苹果与欧盟达成了新的合作协议。"],
        "特斯拉公司": ["正式停产 Model S/Model X。", "全新 Roadster 超跑即将发布。"],
        "谷歌公司": ["Gemma 4 开源大模型全系列发布。", "Veo 3.1 AI 视频生成全面开放。"]
    }
    new_list = mock_news.get(company, [f"未找到 {company} 的新闻数据"])
    return "\n".join(new_list)

# 创建 Agent，调用工具
agent = create_agent(
    model=deepseek_llm,
    tools = [get_stock_price, search_news],
)

resp = agent.invoke({"messages": [{"role": "user", "content": "苹果公司股票价格"}]})
print(resp)
print(resp["messages"][-1].content)
