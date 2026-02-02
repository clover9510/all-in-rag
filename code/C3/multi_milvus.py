import os
from tqdm import tqdm
from glob import glob
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
import numpy as np
import cv2
from PIL import Image

# 1. 初始化设置
MODEL_NAME = "BAAI/bge-base-en-v1.5"
MODEL_PATH = "../../models/bge/Visualized_base_en_v1.5.pth"
DATA_DIR = "../../data/C3"
COLLECTION_NAME = "multimodal_demo"
MILVUS_URI = "http://localhost:19530"


# 2. 定义工具 (编码器和可视化函数)
# 加载一个预训练的多模态模型
class Encoder:
    """编码器类，用于将图像和文本编码为向量。"""

    def __init__(self, model_name: str, model_path: str):
        self.model = Visualized_BGE(model_name_bge=model_name, model_weight=model_path)
        self.model.eval()

    def encode_query(self, image_path: str, text: str) -> list[float]:
        with torch.no_grad():
            #文本、图像联合编码
            query_emb = self.model.encode(image=image_path, text=text)
        return query_emb.tolist()[0]

    def encode_image(self, image_path: str) -> list[float]:
        with torch.no_grad():
            query_emb = self.model.encode(image=image_path)
        return query_emb.tolist()[0]


def visualize_results(query_image_path: str, retrieved_images: list, img_height: int = 300, img_width: int = 300,
                      row_count: int = 3) -> np.ndarray:
    """功能：从检索到的图像列表中创建一个全景图，用于可视化查询结果。
        参数说明：
        query_image_path: 查询图像的路径（字符串类型）。
        retrieved_images: 检索到的图像路径列表（列表类型）。
        img_height: 每张图像的高度，默认为 300 像素。
        img_width: 每张图像的宽度，默认为 300 像素。
        row_count: 每行显示的图像数量，默认为 3。
        返回值：一个 NumPy 数组，表示生成的全景图。"""

    #  创建画布
    # 计算主画布尺寸
    panoramic_width = img_width * row_count #  宽度 = 单张图片宽度 × 每行张数
    panoramic_height = img_height * row_count # 高度 = 单张图片高度 × 行数（自动计算）
    # panoramic_image：用于放置检索到的图像，尺寸为(panoramic_height, panoramic_width)。
    panoramic_image = np.full((panoramic_height, panoramic_width, 3), 255, dtype=np.uint8)
    # query_display_area：专门用于显示查询图像，宽度固定为img_width，高度与主画布一致
    query_display_area = np.full((panoramic_height, img_width, 3), 255, dtype=np.uint8)

    # 处理查询图像
    # 打开查询图像并转换为RGB格式
    query_pil = Image.open(query_image_path).convert("RGB")
    # 转为NumPy数组，然后进行颜色通道转换（RGB→BGR）; [:, :, ::-1] 是切片操作，把RGB变成BGR顺序（OpenCV需要）
    query_cv = np.array(query_pil)[:, :, ::-1]
    # 调整图片大小到指定尺寸
    resized_query = cv2.resize(query_cv, (img_width, img_height))
    # 给图像添加红色边框（cv2.copyMakeBorder）。
    bordered_query = cv2.copyMakeBorder(resized_query, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(255, 0, 0))
    # 将处理后的查询图像放置在 query_display_area 的底部区域。
    query_display_area[img_height * (row_count - 1):, :] = cv2.resize(bordered_query, (img_width, img_height))
    # 在图像上添加文字标注 "Query"。
    cv2.putText(query_display_area, "Query", (10, panoramic_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 处理检索到的图像
    for i, img_path in enumerate(retrieved_images):
        #计算当前图像在全景图中的位置（行号 row 和列号 col）。
        row, col = i // row_count, i % row_count
        #计算图片在画布上的起始坐标
        start_row, start_col = row * img_height, col * img_width
        #打开图像并转换为 RGB 格式，再转为 OpenCV 格式。
        retrieved_pil = Image.open(img_path).convert("RGB")
        retrieved_cv = np.array(retrieved_pil)[:, :, ::-1]
        #调整大小，但比画布的格子小4像素（为了留边框位置）
        resized_retrieved = cv2.resize(retrieved_cv, (img_width - 4, img_height - 4))
        #加2像素的黑色边框,这样总大小就是(img_width, img_height)
        bordered_retrieved = cv2.copyMakeBorder(resized_retrieved, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        #将处理后的图像粘贴到 panoramic_image 的对应位置。
        panoramic_image[start_row:start_row + img_height, start_col:start_col + img_width] = bordered_retrieved
        # 添加索引号
        cv2.putText(panoramic_image, str(i), (start_col + 10, start_row + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    2)
    #将查询区域和主画布水平拼接
    return np.hstack([query_display_area, panoramic_image])


# 3. 初始化客户端
print("--> 正在初始化编码器和Milvus客户端...")
encoder = Encoder(MODEL_NAME, MODEL_PATH)
milvus_client = MilvusClient(uri=MILVUS_URI)

# 4. 创建 Milvus Collection
print(f"\n--> 正在创建 Collection '{COLLECTION_NAME}'")
if milvus_client.has_collection(COLLECTION_NAME):
    milvus_client.drop_collection(COLLECTION_NAME)
    print(f"已删除已存在的 Collection: '{COLLECTION_NAME}'")

image_list = glob(os.path.join(DATA_DIR, "dragon", "*.png"))
if not image_list:
    raise FileNotFoundError(f"在 {DATA_DIR}/dragon/ 中未找到任何 .png 图像。")
dim = len(encoder.encode_image(image_list[0]))

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),      #浮点数向量字段 vector，用于存储图像特征向量
    FieldSchema(name="image_path", dtype=DataType.VARCHAR, max_length=512),#字符串字段 image_path，用于存储对应的图像路径
]

# 创建集合 Schema
schema = CollectionSchema(fields, description="多模态图文检索")
print("Schema 结构:")
print(schema)

# 创建集合
milvus_client.create_collection(collection_name=COLLECTION_NAME, schema=schema)
print(f"成功创建 Collection: '{COLLECTION_NAME}'")
print("Collection 结构:")
print(milvus_client.describe_collection(collection_name=COLLECTION_NAME))

# --> 正在创建 Collection 'multimodal_demo'
#
# Schema 结构:
# {
#     'auto_id': True,
#     'description': '多模态图文检索',
#     'fields': [
#         {'name': 'id', 'description': '', 'type': <DataType.INT64: 5>, 'is_primary': True, 'auto_id': True},
#         {'name': 'vector', 'description': '', 'type': <DataType.FLOAT_VECTOR: 101>, 'params': {'dim': 768}},
#         {'name': 'image_path', 'description': '', 'type': <DataType.VARCHAR: 21>, 'params': {'max_length': 512}}
#     ],
#     'enable_dynamic_field': False
# }
#
# 成功创建 Collection: 'multimodal_demo'
#
# Collection 结构:
# {
#     'collection_name': 'multimodal_demo',
#     'auto_id': True,
#     'num_shards': 1,
#     'description': '多模态图文检索',
#     'fields': [
#         {'field_id': 100, 'name': 'id', 'description': '', 'type': <DataType.INT64: 5>, 'params': {}, 'auto_id': True, 'is_primary': True},
#         {'field_id': 101, 'name': 'vector', 'description': '', 'type': <DataType.FLOAT_VECTOR: 101>, 'params': {'dim': 768}},
#         {'field_id': 102, 'name': 'image_path', 'description': '', 'type': <DataType.VARCHAR: 21>, 'params': {'max_length': 512}}
#     ],
#     'functions': [],
#     'aliases': [],
#     'collection_id': 459243798405253751,
#     'consistency_level': 2,
#     'properties': {},
#     'num_partitions': 1,
#     'enable_dynamic_field': False,
#     'created_timestamp': 459249546649403396,
#     'update_timestamp': 459249546649403396
# }
# 5. 准备并插入数据
print(f"\n--> 正在向 '{COLLECTION_NAME}' 插入数据")
data_to_insert = []
for image_path in tqdm(image_list, desc="生成图像嵌入"):
    vector = encoder.encode_image(image_path)
    data_to_insert.append({"vector": vector, "image_path": image_path})

if data_to_insert:
    result = milvus_client.insert(collection_name=COLLECTION_NAME, data=data_to_insert)
    print(f"成功插入 {result['insert_count']} 条数据。")


# 6. 创建索引
print(f"\n--> 正在为 '{COLLECTION_NAME}' 创建索引")
index_params = milvus_client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    index_type="HNSW",
    metric_type="COSINE",
    params={"M": 16, "efConstruction": 256}
)
milvus_client.create_index(collection_name=COLLECTION_NAME, index_params=index_params)
print("成功为向量字段创建 HNSW 索引。")
print("索引详情:")
print(milvus_client.describe_index(collection_name=COLLECTION_NAME, index_name="vector"))
milvus_client.load_collection(collection_name=COLLECTION_NAME)
print("已加载 Collection 到内存中。")
# --> 正在为 'multimodal_demo' 创建索引
# 成功为向量字段创建 HNSW 索引。
# 索引详情:
# {'M': '16', 'efConstruction': '256', 'metric_type': 'COSINE', 'index_type': 'HNSW', 'field_name': 'vector', 'index_name': 'vector', 'total_rows': 0, 'indexed_rows': 0, 'pending_index_rows': 0, 'state': 'Finished'}
# 已加载 Collection 到内存中。

# 7. 执行多模态检索
print(f"\n--> 正在 '{COLLECTION_NAME}' 中执行检索")
query_image_path = os.path.join(DATA_DIR, "dragon", "query.png")
query_text = "一条龙"
query_vector = encoder.encode_query(image_path=query_image_path, text=query_text)

search_results = milvus_client.search(
    collection_name=COLLECTION_NAME,
    data=[query_vector],
    output_fields=["image_path"],
    limit=5,
    search_params={"metric_type": "COSINE", "params": {"ef": 128}}
)[0]

retrieved_images = []
print("检索结果:")
for i, hit in enumerate(search_results):
    print(f"  Top {i+1}: ID={hit['id']}, 距离={hit['distance']:.4f}, 路径='{hit['entity']['image_path']}'")
    retrieved_images.append(hit['entity']['image_path'])

# --> 正在 'multimodal_demo' 中执行检索
# 检索结果:
#   Top 1: ID=459243798403756667, 距离=0.9411, 路径='../../data/C3\dragon\dragon01.png'
#   Top 2: ID=459243798403756668, 距离=0.5818, 路径='../../data/C3\dragon\dragon02.png'
#   Top 3: ID=459243798403756671, 距离=0.5731, 路径='../../data/C3\dragon\dragon05.png'
#   Top 4: ID=459243798403756670, 距离=0.4894, 路径='../../data/C3\dragon\dragon04.png'
#   Top 5: ID=459243798403756669, 距离=0.4100, 路径='../../data/C3\dragon\dragon03.png'


# 8. 可视化与清理
print(f"\n--> 正在可视化结果并清理资源")
if not retrieved_images:
    print("没有检索到任何图像。")
else:
    panoramic_image = visualize_results(query_image_path, retrieved_images)
    combined_image_path = os.path.join(DATA_DIR, "search_result.png")
    cv2.imwrite(combined_image_path, panoramic_image)
    print(f"结果图像已保存到: {combined_image_path}")
    Image.open(combined_image_path).show()

milvus_client.release_collection(collection_name=COLLECTION_NAME)
print(f"已从内存中释放 Collection: '{COLLECTION_NAME}'")
milvus_client.drop_collection(COLLECTION_NAME)
print(f"已删除 Collection: '{COLLECTION_NAME}'")




