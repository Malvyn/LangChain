from langchain_classic.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_classic.chains.llm import LLMChain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 创建model
model = ChatDeepSeek(model_name="deepseek-chat", temperature=0)

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

# 第一步: 切割阶段
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)

# ── Map 链 ────────────────────────────────────────────────────────────────────
map_template = """
以下是一组文档(document)
"{docs}"
根据这个文档列表，请给出总结摘要: """
map_prompt = PromptTemplate.from_template(map_template)
map_chain = map_prompt | model | StrOutputParser()

# ── Reduce 链 ─────────────────────────────────────────────────────────────────
reduce_template = """
以下是一组总结摘要
"{docs}"
将这些内容提炼成一个最终的、统一的总结摘要 """
reduce_prompt = PromptTemplate.from_template(reduce_template)
reduce_chain = reduce_prompt | model | StrOutputParser()

# ── 替代 ReduceDocumentsChain + MapReduceDocumentsChain ───────────────────────
TOKEN_MAX = 4000

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def collapse_summaries(texts: list[str]) -> str:
    """复现 ReduceDocumentsChain 折叠逻辑：超出 token_max 时分批递归 collapse"""
    combined = "\n\n".join(texts)
    if estimate_tokens(combined) <= TOKEN_MAX:
        return reduce_chain.invoke({"docs": combined})

    # 超出限制：按 TOKEN_MAX 分批
    batches, current_batch, current_tokens = [], [], 0
    for text in texts:
        t = estimate_tokens(text)
        if current_tokens + t > TOKEN_MAX and current_batch:
            batches.append(current_batch)
            current_batch, current_tokens = [], 0
        current_batch.append(text)
        current_tokens += t
    if current_batch:
        batches.append(current_batch)

    # 每批 collapse 后递归
    collapsed = [reduce_chain.invoke({"docs": "\n\n".join(batch)}) for batch in batches]
    return collapse_summaries(collapsed)

# ── Map 阶段：对每个 chunk 单独总结 ───────────────────────────────────────────
summaries = [map_chain.invoke({"docs": doc.page_content}) for doc in split_docs]

# ── Reduce 阶段：递归折叠 + 最终合并 ──────────────────────────────────────────
output_text = collapse_summaries(summaries)
print(output_text)