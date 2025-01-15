import os
import faiss
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import re

# 1. 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 初始化向量化模型
embedding_model_path = os.path.join("models", "shibing624_text2vec-base-chinese")
# rag矢量文件路径
index_path = os.path.join("faiss_index", "faiss_index.bin")
# rag文件原始版
index_source = os.path.join("faiss_index", "jiudian.txt")

# model_path = os.path.join("models", "Qwen2.5-3B-Instruct")    # 替换为你的模型路径
model_path = os.path.join("models", "NousResearch_Hermes-3-Llama-3.2-3B")  # 替换为你的模型路径

embedding_model = SentenceTransformer(embedding_model_path).to(device)  # 替换为适合中文的模型
# 加载 FAISS 索引
index = faiss.read_index(index_path)

# 读取知识库（与生成索引时的文本一一对应）
with open(index_source, "r", encoding="utf-8") as f:
    knowledge_base = [line.strip() for line in f.readlines()]


# 查询函数
def retrieve_top_k(query, k=5):
    # 将输入转为向量
    query_vector = embedding_model.encode([query], convert_to_numpy=True)

    # 检索 top-k
    distances, indices = index.search(query_vector, k)
    results = [{"text": knowledge_base[i], "distance": distances[0][j]} for j, i in enumerate(indices[0])]
    return results


# 测试查询
# query = "中国的首都是哪里？"
# retrieved_docs = retrieve_top_k(query, k=3)
#
# print("检索结果：")
# for doc in retrieved_docs:
#     print(doc["text"])



# 加载量化后的生成模型（GGUF 需要支持的引擎加载，这里以 transformers 为例）

# 2. 加载中文模型（以 Qwen2.5-3B-Instruct 为例）
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# RAG 查询接口
def rag_generate(query, k=3):
    # 检索知识条目
    retrieved_docs = retrieve_top_k(query, k=k)
    knowledge_context = "\n".join([doc["text"] for doc in retrieved_docs])

    # 拼接上下文和用户问题
    # input_text = f"知识上下文：{knowledge_context}\n用户问题：{query}\n回答："
    input_text = f"知识上下文：{knowledge_context}\n用户问题：{query}\n回答："

    # 检查 pad_token，如果未设置，设置为 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 编码输入
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to("cuda")

    # 模型生成
    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],  # 显式传入 attention_mask
        max_length=1000,
        num_return_sequences=1,  # 仅返回一个回答
        do_sample=True,  # 开启采样， 确保生成结果是最可能的一个
    )

    # 解码输出
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 正则匹配用户问题及其对应回答
    pattern = f"(用户问题：{re.escape(query)}[\\s\\S]*?回答：.*?。)"
    match = re.search(pattern, answer)

    if match:
        answer = match.group(1)  # 提取匹配的部分
    else:
        answer = "未找到匹配的回答"

    print(answer)
    return answer



# 测试 RAG
# query = "长城在哪里？"
# answer = rag_generate(query, k=3)
# print("回答：", answer)
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()

# 查询模型
class Query(BaseModel):
    question: str

@app.post("/rag/")
async def rag(query: Query):
    answer = rag_generate(query.question, k=3)
    return {"answer": answer}

# uvicorn app:app --host 0.0.0.0 --port 8000

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)