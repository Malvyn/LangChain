from calendar import error

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, dynamic_prompt, wrap_tool_call
from langchain.chat_models import init_chat_model
from langchain.tools import tool

from env_utils import DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY, DASHSCOPE_API_URL, \
    DASHSCOPE_API_KEY
from my_llm import deepseek_llm
from langchain_core.messages import HumanMessage, ToolMessage
from typing import Callable

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
    raise ValueError("股票接口不可用")
    # mock_data = {
    #     "苹果公司": {"today": 185.20, "week": 183.50, "month": 182.00, "year": 175.00},
    #     "特斯拉公司": {"today": 240.50, "week": 238.00, "month": 235.00, "year": 220.00},
    #     "谷歌公司": {"today": 142.50, "week": 141.00, "month": 139.50, "year": 130.00},
    # }
    #
    # # 标准化公司名称 - 关键修改
    # normalized_company = company_mapping.get(company, company)
    #
    # if normalized_company in mock_data:
    #     price = mock_data[normalized_company].get(timeframe, "未找到该时间范围的价格数据")
    #     return f"{normalized_company} {timeframe}价格: {price}美元"
    # else:
    #     return f"未找到 {company} 的股票数据"


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

basic_mode = init_chat_model(
    model="deepseek-chat",
    model_provider="deepseek",
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

advance_mode = init_chat_model(
    model="qwen-plus",
    model_provider="openai",
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_API_URL,
)

@wrap_model_call
def dynamic_model_selection(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """根据用户输入动态选择模型调用"""

    message_count = len(request.state["messages"])

    if message_count >=3:
        model = advance_mode
    else:
        model = basic_mode

    return handler(request.override(model=model))

@dynamic_prompt
def dynamic_prompt(request: ModelRequest) -> str:
    """根据用户的类型来使用不同的提示词"""
    print("request", request)
    user_type = "normal"

    if request.runtime and request.runtime.context:
        user_type = request.runtime.context.get("user_type", "normal")

    if user_type == "vip":
        prompt = "回答用户问题之前，首先称呼：尊贵的vip客户你好，然后再回答用户问题"
    else:
        prompt = "直接回答用户问题"

    return prompt

@wrap_tool_call
def handle_tool_error(request,handler):
    try:
       return handler(request)
    except Exception as e:
        return ToolMessage(
            tool_call_id=request.tool_call["id"],
            content=f"目前工具服务不可用：{str(e)}"
            )


# 创建 Agent
agent = create_agent(
    model=basic_mode,
    tools=[get_stock_price, search_news],
    middleware=[dynamic_model_selection,dynamic_prompt,handle_tool_error],
)

# 调用 Agent
response = agent.invoke({"messages": [{"role": "user", "content": "查找谷歌公司的股价和新闻"}]},
                        context ={"user_type": "xxx"}
                        )
print(response)
print(response["messages"][-1].content)
