from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek

# 创建model
model = ChatDeepSeek(model_name="deepseek-chat", temperature=0)

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

prompt_template = """
针对下面的内容，写一个简洁的总结摘要：
文章内容: {context}
简洁的总结摘要: """
prompt = PromptTemplate.from_template(prompt_template)


chain = create_stuff_documents_chain(model, prompt)

result = chain.invoke({"context": docs})
print(result)