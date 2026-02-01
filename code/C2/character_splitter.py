"""固定分块test"""
from langchain.text_splitter import  CharacterTextSplitter
from langchain_community.document_loaders import TextLoader

data_path = "../../data/C2/txt/蜂医.txt"

loader=TextLoader(data_path, encoding="utf-8")
#loader 是 TextLoader 类的实例，用于加载纯文本文件；loader.load()会读取 data_path 指向的文件（即 "../../data/C2/txt/蜂医.txt"），按 UTF-8 编码解析内容
docs = loader.load()

text_splitter=CharacterTextSplitter(
    chunk_size=200,    # 每个块的目标大小为100个字符
    chunk_overlap=10   # 每个块之间重叠10个字符，以缓解语义割裂
)
chunks =text_splitter.split_documents(docs)
print(f"文本被切分为 {len(chunks)} 个块。\n")
print("--- 前5个块内容示例 ---")
for i, chunk in enumerate(chunks[:5]):
    print("=" * 60)
    """
    示例:
    块 1 (长度: 72): 
    "page_content='# 蜂医

    游戏《三角洲行动》中的支援型干员
    
    蜂医是2024年琳琅天上发行的《三角洲行动》中的支援型干员之一，在早期版本是唯一一个支援型干员。' metadata={'source': '../../data/C2/txt/蜂医.txt'}"
    ============================================================
    块 1 (长度: 72): "# 蜂医
    
    游戏《三角洲行动》中的支援型干员
    
    蜂医是2024年琳琅天上发行的《三角洲行动》中的支援型干员之一，在早期版本是唯一一个支援型干员。"
   """
    print(f'块 {i + 1} (长度: {len(chunk.page_content)}): "{chunk}"')
    print("=" * 60)
    """
    """
    print(f'块 {i+1} (长度: {len(chunk.page_content)}): "{chunk.page_content}"')