from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document


texts = [
    "张三是法外狂徒",
    "FAISS是一个用于高效相似性搜索和密集向量聚类的库。",
    "LangChain是一个用于开发由语言模型驱动的应用程序的框架。"
]

#1.文档和嵌入模型
docs=[Document(page_content=t) for t in texts]
embeddings=HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

#2.创建向量存储并保存到本地
vectorstore=FAISS.from_documents(docs,embeddings)
local_faiss_path = "./faiss_index_store"
vectorstore.save_local(local_faiss_path)
print(f"FAISS index has been saved to {local_faiss_path}")

# 3. 加载索引并执行查询
loaded_vectorstore =vectorstore.load_local(local_faiss_path,embeddings,allow_dangerous_deserialization=True)

# 4. 相似性搜索
query = "FAISS是做什么的？"
results=loaded_vectorstore.similarity_search(query,k=1)
print(f"\n查询: '{query}'")
print("相似度最高的文档:")
for doc in results:
    '''
    详细- page_content='FAISS是一个用于高效相似性搜索和密集向量聚类的库。'
        - FAISS是一个用于高效相似性搜索和密集向量聚类的库。'''
    print(f"详细- {doc}")
    print(f"- {doc.page_content}")