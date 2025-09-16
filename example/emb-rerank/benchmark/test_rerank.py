import time
import requests
from typing import List

from transformers import AutoTokenizer

def calculate_rerank_input_length(query: str, documents: List[str], tokenizer):
    """
    计算 rerank 任务的输入长度
    通常是 query + document 的总 token 数量
    """
    input_lengths = []
    
    for doc in documents:
        # 通常的格式: [CLS] query [SEP] document [SEP]
        combined_text = f"{query} {doc}"
        tokens = tokenizer.encode(combined_text, add_special_tokens=True)
        input_lengths.append(len(tokens))
    
    return sum(input_lengths)

# 使用示例


def tei_rank(iterations: int = 1, batch_size_5: int = 1):
    text_1 = "兵马俑，位于中国陕西省西安市临潼区，是秦始皇陵的一部分，被誉为'世界第八大奇迹'。这些陶俑始建于公元前246年，历时38年完成，目的是为了在秦始皇死后守护他的陵墓。兵马俑坑分为三个主要部分，分别是一号坑、二号坑和三号坑，其中一号坑规模最大，展示了数千名士兵、战马和战车的壮观阵容。每个陶俑的面部特征、表情和服饰都各不相同，栩栩如生，展现了古代工匠高超的技艺。兵马俑不仅是秦朝军事力量的象征，也是中国古代雕塑艺术的巅峰之作。它们的发现为研究秦代历史、文化和军事提供了宝贵的实物资料，吸引了无数游客和学者前来参观和研究。兵马俑的存在，见证了中国古代文明的辉煌与伟大。"

    texts_2 = [
        "兵马俑作为秦始皇陵的重要组成部分，不仅是中国古代陵墓文化的杰出代表，也是世界考古史上的重大发现。这些陶俑的制作始于公元前246年，历时38年完成，规模宏大，工艺精湛。兵马俑的发现为研究秦代的军事制度、武器装备和社会结构提供了珍贵的实物资料。每个陶俑的面部特征和服饰细节都经过精心雕刻，展现了古代工匠的高超技艺。兵马俑的存在不仅体现了秦始皇对死后世界的重视，也反映了秦朝强大的国力和组织能力。如今，兵马俑已成为中国文化的象征，吸引了全球无数游客和学者前来参观和研究。",
        "兵马俑的艺术价值在于其独特的写实风格和精湛的制作工艺。每个陶俑的面部表情、发型和服饰都各不相同，栩栩如生，仿佛再现了秦朝军队的真实面貌。陶俑的身高、体型和姿态也根据其军衔和职责进行了细致的设计，展现了古代工匠对细节的极致追求。此外，兵马俑的彩绘工艺也令人惊叹，虽然大部分颜色已随时间褪去，但部分陶俑仍保留了鲜艳的彩绘痕迹。这些陶俑不仅是秦朝军事力量的象征，更是中国古代雕塑艺术的巅峰之作。兵马俑的发现，为研究古代艺术史提供了宝贵的实物资料。",
        "现代科技的飞速发展正在深刻改变人类的生活方式。人工智能、大数据和物联网等技术的广泛应用，使得生产效率大幅提升，同时也催生了新的产业和商业模式。例如，自动驾驶技术的成熟正在重塑交通运输行业，而虚拟现实技术则为娱乐和教育领域带来了全新的体验。此外，量子计算和生物技术的突破也为解决全球性问题（如气候变化和疾病治疗）提供了新的可能性。尽管科技发展带来了诸多便利，但也引发了隐私保护、伦理道德等方面的挑战。如何在科技进步与社会责任之间找到平衡，是未来需要面对的重要课题。",
        "兵马俑的考古工作自1974年被发现以来，一直是全球考古学界的焦点。考古学家通过对一号坑、二号坑和三号坑的系统发掘，逐步揭示了秦代军事装备、军队编制以及陶俑制作工艺的细节。研究发现，兵马俑的排列并非随意，而是严格按照秦代军队的实战阵型布置，前锋、中军和后卫分工明确。陶俑的制作采用了模块化生产方式，不同身体部位分别烧制后组装，体现了秦代高效的组织能力。此外，兵马俑的彩绘技术尤为精湛。考古人员发现，陶俑原本通体施彩，颜料以矿物质为主，历经两千余年仍部分保留鲜艳色泽。近年来，中德合作团队利用3D扫描和光谱分析技术，成功复原了部分陶俑的原始色彩，为研究秦代服饰文化提供了重要依据。由于氧化和湿度影响，兵马俑的保护面临巨大挑战。目前，文物保护专家采用纳米材料加固、控温控湿等先进技术延缓陶俑的退化。同时，数字化存档和虚拟复原技术也被应用于兵马俑的长期保存与研究。这些努力不仅让这一世界文化遗产得以延续，也为其他类似文物的保护提供了借鉴。",
        "随着城市化进程加速，城市绿化已成为提升居民生活质量的关键举措。许多国际大都市通过建设垂直花园、屋顶绿地和生态走廊，有效缓解了热岛效应，并改善了空气质量。例如，新加坡的'花园城市'计划通过立法要求新建建筑必须包含绿化面积，其滨海湾花园的超级树景观更是成为全球典范。绿色基础设施不仅美化环境，还具有显著的生态效益。树木和植被能吸收二氧化碳、降低噪音，并为野生动物提供栖息地。近年来，智能灌溉系统和耐旱植物的推广进一步提高了绿化项目的可持续性。巴黎的'城市森林'计划甚至提出到2030年将50%的市区面积转化为绿地，以应对气候变化。然而，城市绿化也面临土地资源有限、维护成本高等挑战。一些城市尝试通过社区参与和公私合作模式解决这些问题，如纽约的'百万树木计划'便动员了数千名志愿者参与种植。未来，结合生物多样性和低碳技术的绿化设计将成为城市发展的核心方向。"
    ]* batch_size_5
    tokenizer = AutoTokenizer.from_pretrained("/home/tonyguo/emb_models/reranker/bge-reranker-v2-m3_tvm0728_512/bge-reranker-v2-m3/tokenizer")
    lengths = calculate_rerank_input_length(text_1, texts_2, tokenizer)
    print("每个 [query, doc] 对的 token 数量:", lengths)

    # TEI 服务配置
    TEI_URL = "http://localhost:9999/v1/rerank"  # 注意使用/rerank端点
    headers = {"Content-Type": "application/json"}

    # 准备请求数据 (格式需符合TEI的rerank API要求)
    pairs = [{"query": text_1, "text": text} for text in texts_2]

    # Warm-up (if iterations > 1)
    if iterations > 1:
        print("Running warm-up...")
        for _ in range(10):
            requests.post(
                TEI_URL,
                headers=headers,
                json={"query": text_1, "model": "rerank_vacc", "documents": texts_2[:1], "return_documents": True}  # 单条预热
            )

    # Run inference and measure latency
    latencies = []
    for i in range(iterations):
        start_time = time.perf_counter()

        response = requests.post(
            TEI_URL,
            headers=headers,
            json={"query": text_1, "documents": texts_2, "model": "rerank_vacc", "return_documents": True}
        )
        latency = time.perf_counter() - start_time
        latencies.append(latency)

        print(f"Iteration {i+1}/{iterations} | Latency: {latency:.4f}s")

        # Print scores only for last iteration
        # if i == (iterations - 1):
        #     score_items = response.json()  # 获取带index的分数列表
        #     # 按原始texts_2顺序提取分数（需根据index映射）
        #     sorted_scores = sorted(score_items, key=lambda x: x['index'])
        #     scores = [item['score'] for item in sorted_scores]

        #     # 打印结果（保持与输入顺序一致）
        #     for text, score in zip(texts_2, scores):
        #         print(f"Pair: {[text_1[:50] + '...', text[:50] + '...']} | Score: {score:.4f}")

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
    tei_rank(iterations=1, batch_size_5=1)