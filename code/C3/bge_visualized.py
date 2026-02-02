"""多模态图文"""
import torch
from visual_bge.visual_bge.modeling import Visualized_BGE
model = Visualized_BGE(model_name_bge="BAAI/bge-base-en-v1.5",# 基础文本编码器名称（BAAI的大规模预训练模型）
                      model_weight="../../models/bge/Visualized_base_en_v1.5.pth") # 预训练权重文件路径

#进入评估模式（不进行训练）
model.eval()
with torch.no_grad():  # 不计算梯度，节省内存和计算资源
    text_emb = model.encode(text="blue whale")  # 纯文本的向量
    img_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png")  # 第一张图的向量
    multi_emb_1 = model.encode(image="../../data/C3/imgs/datawhale01.png", text="blue whale")  # 建立了图片和文字的直接联系
    img_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png")  # 第二张图的向量
    multi_emb_2 = model.encode(image="../../data/C3/imgs/datawhale02.png", text="datawhale开源组织的logo")  # 另一组图+文组合向量
    # 编码纯文本"猫"
    text_emb_2 = model.encode(text="猫")  # 纯文本的向量
    multi_emb_3 = model.encode(image="../../data/C3/imgs/cat.png", text="猫")  # 建立图片和文字的直接联系,同时给他看猫的图片 + 说"这是猫"
    multi_emb_4 = model.encode(image="../../data/C3/imgs/cat.jpg", text="猫")  # 图+文组合向量
    img_emb_3 = model.encode(image="../../data/C3/imgs/cat.jpg")  # 第一张图的向量
    img_emb_4 = model.encode(image="../../data/C3/imgs/cat.png")  # 第一张图的向量


# 计算相似度（点积运算）
sim_1 = img_emb_1 @ img_emb_2.T  # 两张datawhale logo图片之间的相似度
sim_2 = img_emb_2 @ multi_emb_1.T  # 纯图片 vs 图片+文本1组合的相似度
sim_3 = text_emb @ multi_emb_1.T  # 纯文本 vs 图片+文本组合的相似度
sim_4 = multi_emb_1 @ multi_emb_2.T  # 两个datawhale图文组合之间的相似度
sim_5 = multi_emb_3 @ multi_emb_4.T  # 两个猫的图文组合之间的相似度
sim_6 = img_emb_4 @ multi_emb_4.T  # 猫图、图文组合之间的相似度
sim_7 = img_emb_4 @ img_emb_3.T  # 两个猫的图之间的相似度
sim_8 =multi_emb_3 @ text_emb_2.T  # 两个猫的图之间的相似度
# 相似度范围：通常-1到1之间，越接近1越相似，越接近-1越不相似
print("=== 相似度计算结果 ===")
print("=== 相似度计算结果 ===")
print(f"纯图像 vs 纯图像: {sim_1}")  # 应该是两个相似的logo，相似度应该较高
print(f"图文结合1 vs 纯图像: {sim_2}")  # 同一张图片的不同表示
print(f"图文结合1 vs 纯文本: {sim_3}")  # 文字描述与图文组合的相似度
print(f"图文结合1 vs 图文结合2: {sim_4}")  # 两个不同图片但相同文本的组合
print(f"两个猫图文组合的相似度: {sim_5}") #tensor([[0.8484]])
print(f"猫图、图文组合的相似度: {sim_6}") #tensor([[0.8307]])
print(f"两个猫图的相似度: {sim_7}") #tensor([[0.8407]])
print(f"猫文本、猫图组合的相似度: {sim_8}") #tensor([[0.5040]])
# 纯文本空间 <---0.5040---> 多模态空间
#       |                         |
#       |0.8307(估计)             |0.8484
#       |                         |
# 纯图像空间 <---0.8407---> 纯图像空间
# 从你的结果可以看到：
#
# 视觉-视觉相似度最高 (0.8407)
#
# 两张猫图之间最相似
#
# 多模态-多模态相似度高 (0.8484)
#
# 两个猫的图文组合很相似
#
# 多模态-纯文本相似度中等 (0.5040)
#
# 图文组合与纯文本有一定相关性，但不是最高
#
# 纯图像-多模态相似度高 (0.8307)
#
# 猫图与猫图文组合很相似
#
# 这说明多模态表示很大程度上保留了视觉信息