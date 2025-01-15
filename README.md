# 基于hugging face的向量化数据,以及本地加载模型服务

## 项目所有模型需要自己去hugging face 下载!!!!!!
[NousResearch_Hermes-3-Llama-3.2-3B] https://huggingface.co/NousResearch/Hermes-3-Llama-3.2-3B
[Qwen2.5-3B-Instruct] https://huggingface.co/Qwen/Qwen2.5-3B-Instruct
[shibing624_text2vec-base-chinese] https://huggingface.co/shibing624/text2vec-base-chinese
项目根目录包含以下内容：
- README.md: 项目的说明文档。
- faiss_index/
  - faiss_index.bin: 向量存储文件。
  - jiudian.txt: rag基础文件。
- models/    本地模型存放位置
  - shibing624_text2vec-base-chinese: embedding models(.safetensors)格式
  - Qwen2.5-3B-Instruct : 从huggingface下载的模型文件(.safetensors)格式
- app.py :主程序,提供接口
- createFaissData.py : 文本转向量数据库

python版本 3.12.5

安装所用库: pip install -r requirements.txt 

## 如果有用可以请我喝咖啡.非常感谢!!

 <img src="https://shitu-query-bj.bj.bcebos.com/2025-01-15/10/631c947a23d31c53?authorization=bce-auth-v1%2F7e22d8caf5af46cc9310f1e3021709f3%2F2025-01-15T02%3A39%3A39Z%2F300%2Fhost%2F3d638e5ce7e9f4d1d67064bf9dbb747feb8a8de9f15752ab6da7722d25b03bb3" width="300" height="400"/>
 <img src="https://shitu-query-bj.bj.bcebos.com/2025-01-15/10/7c4d57b1b22f3c7d?authorization=bce-auth-v1%2F7e22d8caf5af46cc9310f1e3021709f3%2F2025-01-15T02%3A40%3A09Z%2F300%2Fhost%2Fb514f674b321f80e88d38ca2b5b0eb299b1a08f18f10cfcef56ab569aa426c43" width="300" height="400"/>
