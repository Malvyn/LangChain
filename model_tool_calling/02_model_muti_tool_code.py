# 定义股票查询工具
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage  # 1. 导入 ToolMessage

from init_llm import deepseek_llm


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


# 2. 模型绑定工具
model_with_tools = deepseek_llm.bind_tools([get_stock_price, search_news])

# 3. 创建消息列表并添加用户问题
messages = []
humanMessage = HumanMessage(content="苹果公司今天的股票价格是多少？ 有什么新闻")
messages.append(humanMessage)

# 4. 工具调用循环
while True:
    # 调用模型
    response = model_with_tools.invoke(messages)
    # 将模型的响应（可能包含工具调用请求）加入消息历史
    messages.append(response)

    # 检查是否有工具调用
    if response.tool_calls:
        # --- 关键修正：收集所有工具调用的结果 ---
        tool_messages = []
        for tool_call in response.tool_calls:
            print(f"正在调用工具: {tool_call['name']}, 参数: {tool_call['args']}")

            # 执行工具函数
            if tool_call["name"] == "get_stock_price":
                result_content = get_stock_price.func(**tool_call["args"])
            elif tool_call["name"] == "search_news":
                result_content = search_news.func(**tool_call["args"])
            else:
                result_content = f"未知工具: {tool_call['name']}"

            # --- 关键修正：创建 ToolMessage 对象 ---
            # 必须使用 ToolMessage，并传入 tool_call_id 和内容
            tool_message = ToolMessage(content=result_content, tool_call_id=tool_call["id"])
            tool_messages.append(tool_message)

        # --- 关键修正：一次性将所有工具结果加入消息历史 ---
        # 这确保了模型请求和工具响应是正确配对的
        messages.extend(tool_messages)
    else:
        # 如果没有工具调用，说明模型已生成最终答案，跳出循环
        print("没有工具调用，模型已生成最终答案。")
        break

print("\n" + "=" * 50)
print("messages:", messages)
print(response.content)