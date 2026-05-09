from langchain_classic.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_deepseek import ChatDeepSeek
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 创建model
model = ChatDeepSeek(model_name="deepseek-chat", temperature=0)

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

# 第三种：Refine
'''
Refine 是一种基于 MapReduce 文档总结的变体，它在 MapReduce 文档总结的基础上，引入了一种新的文档总结方法。
文档链通过循坏便利输入文档并逐步更新其答案来构建相应。对于每个文档，它将当前文档和最新的的中间答案作为输入，生成一个新的中间答案。   
'''

# 第一步: 切割阶段
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)

# 指定 chain_type 为 refine
chain = load_summarize_chain(model, chain_type="refine")

# Fix: 正确的输入键是 input_documents，不是 input
result = chain.invoke({"input_documents": split_docs})
print(result["output_text"])