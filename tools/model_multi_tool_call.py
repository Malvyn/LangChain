from langchain.tools import tool
from langchain_core.messages import HumanMessage
import yfinance as yf
from my_llm import deepseek_llm


# LLM 使用工具返回结果的步骤
# 1. 定义工具，绑定工具到大模型
# 2. 与LLM进行对话，LLM返回调用大模型的请求，并不会主动调用工具
# 3. 根据返回的结果手动处理，并将结果告知LLM
# 4. LLM 最后生成回复

@tool
def get_stock_price(company: str, timeframe: str = "today") -> str:
    """
    获取股票价格信息。

    Args:
        company: 公司名称或股票代码，如 "苹果", "AAPL", "特斯拉", "TSLA"
        timeframe: 时间范围，today/week/month/year

    Returns:
        股票价格信息
    """
    # 公司名称映射 - 关键修改
    company_mapping = {
        "苹果": "苹果公司",
        "AAPL": "苹果公司",
        "特斯拉": "特斯拉公司",
        "TSLA": "特斯拉公司",
        "谷歌": "谷歌公司",
        "GOOGL": "谷歌公司"
    }

    # 模拟股票数据
    mock_data = {
        "苹果公司": {"today": 185.20, "week": 183.50, "month": 182.00, "year": 175.00},
        "特斯拉公司": {"today": 240.50, "week": 238.00, "month": 235.00, "year": 220.00},
        "谷歌公司": {"today": 142.50, "week": 141.00, "month": 139.50, "year": 130.00},
    }

    # 标准化公司名称 - 关键修改
    normalized_company = company_mapping.get(company, company)

    if normalized_company in mock_data:
        price = mock_data[normalized_company].get(timeframe, "未找到该时间范围的价格数据")
        return f"{normalized_company} {timeframe}价格: {price}美元"
    else:
        return f"未找到 {company} 的股票数据"


@tool
def search_news(company: str) -> str:
    """
    搜索股票新闻。

    Args:
        company: 公司名称或股票代码，如 "苹果", "AAPL", "特斯拉", "TSLA"

    Returns:
        股票新闻信息
    """
    # 公司代码映射 - 关键修改
    company_mapping = {
        "苹果": "AAPL",
        "AAPL": "AAPL",
        "特斯拉": "TSLA",
        "TSLA": "TSLA",
        "谷歌": "GOOGL",
        "GOOGL": "GOOGL"
    }

    # 模拟新闻数据
    mock_news = {
        "AAPL": [
            {"title": "苹果发布新手机", "summary": "新手机功能更加强大"},
            {"title": "苹果股票价格上涨", "summary": "股票价格上涨1%"},
            {"title": "苹果股票价格下跌", "summary": "股票价格下跌1%"},
        ],
        "TSLA": [
            {"title": "特斯拉发布新车", "summary": "新车功能更加强大"},
            {"title": "特斯拉股票价格上涨", "summary": "股票价格上涨1%"},
            {"title": "特斯拉股票价格下跌", "summary": "股票价格下跌1%"},
        ],
        "GOOGL": [
            {"title": "谷歌发布新手机", "summary": "新手机功能更加强大"},
            {"title": "谷歌股票价格上涨", "summary": "股票价格上涨1%"},
            {"title": "谷歌股票价格下跌", "summary": "股票价格下跌1%"},
        ],
    }

    # 标准化公司代码 - 关键修改
    normalized_company = company_mapping.get(company, company)

    if normalized_company in mock_news:
        news_items = mock_news[normalized_company]
        formatted_news = [f"- {item['title']}: {item['summary']}" for item in news_items]
        return f"{company} 的最新新闻:\n" + "\n".join(formatted_news)
    else:
        return f"未找到 {company} 的相关新闻"

# 绑定工具到大模型
model_bind_tool = deepseek_llm.bind_tools([get_stock_price, search_news])

messages = []
#human_message = HumanMessage(content="苹果今日股价和最新新闻")
human_message = HumanMessage(content="比较一下苹果和特斯拉的上周股票价格")
messages.append(human_message)

# 2. LLM返回调用工具的请求
while True:
    resp = model_bind_tool.invoke(messages)
    messages.append(resp)

    if resp.tool_calls:
        for tool_call in resp.tool_calls:
            if tool_call["name"] == "get_stock_price":
                # 3.根据返回的结果手动处理，并将结果告知LLM
                tool_result = get_stock_price.invoke(tool_call)
                messages.append(tool_result)
            elif tool_call["name"] == "search_news":
                # 3.根据返回的结果手动处理，并将结果告知LLM
                tool_result = search_news.invoke(tool_call)
                messages.append(tool_result)

    else:
        print("没有工具调用")
        break

print('message', messages)
print(resp.content)