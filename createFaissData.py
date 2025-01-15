from sentence_transformers import SentenceTransformer
import faiss
import os

# 初始化向量化模型
embedding_model_path = os.path.join("models", "shibing624_text2vec-base-chinese")
embedding_model = SentenceTransformer(embedding_model_path)  # 替换为适合中文的模型
# rag文件原始版
index_source = os.path.join("faiss_index", "jiudian.txt")
# 向量数据库保存位置
save_path = os.path.join("faiss_index", "faiss_index.bin")
# 读取知识库
with open(index_source, "r", encoding="utf-8") as f:
    knowledge_base = [line.strip() for line in f.readlines()]

# 转换为向量
embeddings = embedding_model.encode(knowledge_base, convert_to_numpy=True)

# 创建 FAISS 索引
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# 保存索引到文件
faiss.write_index(index, save_path)

print("FAISS 索引已保存！")
