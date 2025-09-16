import requests

# 定义请求参数
response = requests.post(
    "http://localhost:9998/v1/embeddings",  # Embedding 端点
    json={
        "model": "emb_vacc",  # 替换为你的 Embedding 模型 UID（如 'bge-m3'）
        "input": "A man is eating pasta."    # 支持字符串或字符串列表
    }
)
print("Emb 结果:", response.json())

