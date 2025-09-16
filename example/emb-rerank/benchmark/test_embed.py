import time
import requests
from typing import List

def tei_embed(iterations: int = 1, batch_size: int = 1):
    sentences_short = [
        "人工智能与人类的关系正在经历从简单工具到深度协作的转变，这种变革已经渗透到各个行业的核心环节。"
        "在医疗诊断领域，最新一代的AI辅助系统不仅能够以超过96%的准确率识别医学影像中的微小病灶，更能结合患者的电子病历和基因组数据给出个性化治疗建议。"
        "然而真正具有突破性的是人机协作诊断模式——AI负责处理海量数据，医生则专注于与患者的深度沟通，将冰冷的检测结果转化为温暖的治疗方案。",
    ] * batch_size
    from transformers import AutoTokenizer
    vastai_tokenizer = AutoTokenizer.from_pretrained("/home/tonyguo/emb_models/bge-m3-vacc/tokenizer")
    all_token_nums = sum(len(vastai_tokenizer.tokenize(s)) for s in sentences_short)
    print(f"all_token_nums:{all_token_nums}")
    # TEI 服务配置
    TEI_URL = "http://localhost:9998/v1/embeddings"
    headers = {"Content-Type": "application/json"}

    # Warm-up (if iterations > 1)
    # if iterations > 1:
    #     print("Running warm-up...")
    #     for _ in range(10):
    #         requests.post(
    #             TEI_URL,
    #             headers=headers,
    #             json={"input": ["warm-up"] * len(sentences_short), "model": "emb_vacc"}
    #         )

    # Run inference and measure latency
    latencies = []
    for i in range(iterations):
        start_time = time.perf_counter()
        response = requests.post(
            TEI_URL,
            headers=headers,
            json={"input": sentences_short, "model": "emb_vacc"}
        )
        latency = time.perf_counter() - start_time
        latencies.append(latency)

        print(f"Iteration {i+1}/{iterations} | Latency: {latency:.4f}s")

        # Print embeddings only for first iteration
        # if i == (iterations - 1):
        #     embeddings = np.array(response.json()["data"][0]["embedding"])
        #     print(f"Embeddings shape: {embeddings.shape}")
        #     print(f"Sample embeddings (first 5 dim): {embeddings[:5]}")

    # Statistics summary
    if iterations > 1:
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        print("\n=== Summary ===")
        print(f"Average latency: {avg_latency:.4f}s")
        print(f"Min latency: {min_latency:.4f}s")
        print(f"Max latency: {max_latency:.4f}s")

if __name__ == "__main__":
    tei_embed(iterations=10, batch_size=1)