import requests
response = requests.post(
    "http://localhost:9999/v1/rerank",
    json={
        "model": "rerank_vacc",
        "query": "A man is eating pasta.",
        "documents": [
            "A man is eating food.",
            "A man is eating a piece of bread.",
            "The girl is carrying a baby.",
            "A man is riding a horse.",
            "A woman is playing violin."
        ],
        "return_documents": True
    }
)
print("Rerank 结果:", response.json())

