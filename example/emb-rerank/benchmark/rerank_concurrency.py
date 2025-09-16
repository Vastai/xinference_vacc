import time
import requests
import concurrent.futures
import threading
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
import json

class RerankBenchmark:
    def __init__(self, tei_url: str = "http://localhost:9999/v1/rerank"):
        self.tei_url = tei_url
        self.headers = {"Content-Type": "application/json"}
        self.tokenizer = AutoTokenizer.from_pretrained("/home/tonyguo/emb_models/bge-reranker-v2-m3-vacc/512/tokenizer")
        self.results = {}
    
    def calculate_rerank_input_length(self, query: str, documents: List[str]) -> int:
        """计算 rerank 任务的总 token 数量"""
        total_tokens = 0
        for doc in documents:
            combined_text = f"{query} {doc}"
            tokens = self.tokenizer.encode(combined_text, add_special_tokens=True)
            total_tokens += len(tokens)
        return total_tokens
    
    def generate_rerank_data(self, target_tokens: int) -> Tuple[str, List[str]]:
        """生成指定token长度的rerank测试数据"""
        # 生成query
        query_base = "兵马俑，位于中国陕西省西安市临潼区，是秦始皇陵的一部分，被誉为'世界第八大奇迹'。"
        query_target_tokens = max(10, target_tokens // 10)  # query占1/10，至少10个token
        query = self._extend_text_to_tokens(query_base, query_target_tokens)
        
        # 原始五段不同内容的文档
        original_documents = [
            "兵马俑作为秦始皇陵的重要组成部分，不仅是中国古代陵墓文化的杰出代表，也是世界考古史上的重大发现。这些陶俑的制作始于公元前246年，历时38年完成，规模宏大，工艺精湛。兵马俑的发现为研究秦代的军事制度、武器装备和社会结构提供了珍贵的实物资料。每个陶俑的面部特征和服饰细节都经过精心雕刻，展现了古代工匠的高超技艺。兵马俑的存在不仅体现了秦始皇对死后世界的重视，也反映了秦朝强大的国力和组织能力。如今，兵马俑已成为中国文化的象征，吸引了全球无数游客和学者前来参观和研究。",
            "兵马俑的艺术价值在于其独特的写实风格和精湛的制作工艺。每个陶俑的面部表情、发型和服饰都各不相同，栩栩如生，仿佛再现了秦朝军队的真实面貌。陶俑的身高、体型和姿态也根据其军衔和职责进行了细致的设计，展现了古代工匠对细节的极致追求。此外，兵马俑的彩绘工艺也令人惊叹，虽然大部分颜色已随时间褪去，但部分陶俑仍保留了鲜艳的彩绘痕迹。这些陶俑不仅是秦朝军事力量的象征，更是中国古代雕塑艺术的巅峰之作。兵马俑的发现，为研究古代艺术史提供了宝贵的实物资料。",
            "现代科技的飞速发展正在深刻改变人类的生活方式。人工智能、大数据和物联网等技术的广泛应用，使得生产效率大幅提升，同时也催生了新的产业和商业模式。例如，自动驾驶技术的成熟正在重塑交通运输行业，而虚拟现实技术则为娱乐和教育领域带来了全新的体验。此外，量子计算和生物技术的突破也为解决全球性问题（如气候变化和疾病治疗）提供了新的可能性。尽管科技发展带来了诸多便利，但也引发了隐私保护、伦理道德等方面的挑战。如何在科技进步与社会责任之间找到平衡，是未来需要面对的重要课题。",
            "兵马俑的考古工作自1974年被发现以来，一直是全球考古学界的焦点。考古学家通过对一号坑、二号坑和三号坑的系统发掘，逐步揭示了秦代军事装备、军队编制以及陶俑制作工艺的细节。研究发现，兵马俑的排列并非随意，而是严格按照秦代军队的实战阵型布置，前锋、中军和后卫分工明确。陶俑的制作采用了模块化生产方式，不同身体部位分别烧制后组装，体现了秦代高效的组织能力。此外，兵马俑的彩绘技术尤为精湛。考古人员发现，陶俑原本通体施彩，颜料以矿物质为主，历经两千余年仍部分保留鲜艳色泽。近年来，中德合作团队利用3D扫描和光谱分析技术，成功复原了部分陶俑的原始色彩，为研究秦代服饰文化提供了重要依据。由于氧化和湿度影响，兵马俑的保护面临巨大挑战。目前，文物保护专家采用纳米材料加固、控温控湿等先进技术延缓陶俑的退化。同时，数字化存档和虚拟复原技术也被应用于兵马俑的长期保存与研究。这些努力不仅让这一世界文化遗产得以延续，也为其他类似文物的保护提供了借鉴。",
            "随着城市化进程加速，城市绿化已成为提升居民生活质量的关键举措。许多国际大都市通过建设垂直花园、屋顶绿地和生态走廊，有效缓解了热岛效应，并改善了空气质量。例如，新加坡的'花园城市'计划通过立法要求新建建筑必须包含绿化面积，其滨海湾花园的超级树景观更是成为全球典范。绿色基础设施不仅美化环境，还具有显著的生态效益。树木和植被能吸收二氧化碳、降低噪音，并为野生动物提供栖息地。近年来，智能灌溉系统和耐旱植物的推广进一步提高了绿化项目的可持续性。巴黎的'城市森林'计划甚至提出到2030年将50%的市区面积转化为绿地，以应对气候变化。然而，城市绿化也面临土地资源有限、维护成本高等挑战。一些城市尝试通过社区参与和公私合作模式解决这些问题，如纽约的'百万树木计划'便动员了数千名志愿者参与种植。未来，结合生物多样性和低碳技术的绿化设计将成为城市发展的核心方向。"
        ]

        # 计算query实际占用的tokens
        query_tokens = len(self.tokenizer.encode(query, add_special_tokens=True))
        remaining_tokens = max(0, target_tokens - query_tokens)

        # 平均分配剩余token给5个文档（你也可以按原始长度比例分配）
        doc_target_tokens = remaining_tokens // 10 if remaining_tokens >= 5 else 1

        documents = []
        for i, doc_base in enumerate(original_documents):
            # 为每个文档添加编号前缀（可选，保持一致性）
            extended_doc = self._extend_text_to_tokens(f"{doc_base} 文档{i+1}: ", doc_target_tokens)
            documents.append(extended_doc)

        return query, documents
    
    def _extend_text_to_tokens(self, base_text: str, target_tokens: int) -> str:
        """扩展文本到指定token数量"""
        current_tokens = self.tokenizer.encode(base_text, add_special_tokens=False)
        current_length = len(current_tokens)
        
        if current_length >= target_tokens:
            truncated_tokens = current_tokens[:target_tokens]
            return self.tokenizer.decode(truncated_tokens)
        
        # 添加更多内容
        additional_text = ("。秦始皇陵考古发现古代军事文化雕塑艺术历史研究文物保护科学技术数字化复原" * 
                          (target_tokens // 10 + 1))
        
        full_text = base_text + additional_text
        full_tokens = self.tokenizer.encode(full_text, add_special_tokens=False)
        
        if len(full_tokens) > target_tokens:
            adjusted_tokens = full_tokens[:target_tokens]
            return self.tokenizer.decode(adjusted_tokens)
        else:
            return full_text
    
    def warm_up(self, warm_up_iterations: int = 3):
        """预热Rerank服务"""
        print("Running warm-up...")
        query = "兵马俑，位于中国陕西省西安市临潼区，是秦始皇陵的一部分，被誉为'世界第八大奇迹'。这些陶俑始建于公元前246年，历时38年完成，目的是为了在秦始皇死后守护他的陵墓。兵马俑坑分为三个主要部分，分别是一号坑、二号坑和三号坑，其中一号坑规模最大，展示了数千名士兵、战马和战车的壮观阵容。每个陶俑的面部特征、表情和服饰都各不相同，栩栩如生，展现了古代工匠高超的技艺。兵马俑不仅是秦朝军事力量的象征，也是中国古代雕塑艺术的巅峰之作。它们的发现为研究秦代历史、文化和军事提供了宝贵的实物资料，吸引了无数游客和学者前来参观和研究。兵马俑的存在，见证了中国古代文明的辉煌与伟大。"

        documents = [
            "兵马俑作为秦始皇陵的重要组成部分，不仅是中国古代陵墓文化的杰出代表，也是世界考古史上的重大发现。这些陶俑的制作始于公元前246年，历时38年完成，规模宏大，工艺精湛。兵马俑的发现为研究秦代的军事制度、武器装备和社会结构提供了珍贵的实物资料。每个陶俑的面部特征和服饰细节都经过精心雕刻，展现了古代工匠的高超技艺。兵马俑的存在不仅体现了秦始皇对死后世界的重视，也反映了秦朝强大的国力和组织能力。如今，兵马俑已成为中国文化的象征，吸引了全球无数游客和学者前来参观和研究。",
            "兵马俑的艺术价值在于其独特的写实风格和精湛的制作工艺。每个陶俑的面部表情、发型和服饰都各不相同，栩栩如生，仿佛再现了秦朝军队的真实面貌。陶俑的身高、体型和姿态也根据其军衔和职责进行了细致的设计，展现了古代工匠对细节的极致追求。此外，兵马俑的彩绘工艺也令人惊叹，虽然大部分颜色已随时间褪去，但部分陶俑仍保留了鲜艳的彩绘痕迹。这些陶俑不仅是秦朝军事力量的象征，更是中国古代雕塑艺术的巅峰之作。兵马俑的发现，为研究古代艺术史提供了宝贵的实物资料。",
            "现代科技的飞速发展正在深刻改变人类的生活方式。人工智能、大数据和物联网等技术的广泛应用，使得生产效率大幅提升，同时也催生了新的产业和商业模式。例如，自动驾驶技术的成熟正在重塑交通运输行业，而虚拟现实技术则为娱乐和教育领域带来了全新的体验。此外，量子计算和生物技术的突破也为解决全球性问题（如气候变化和疾病治疗）提供了新的可能性。尽管科技发展带来了诸多便利，但也引发了隐私保护、伦理道德等方面的挑战。如何在科技进步与社会责任之间找到平衡，是未来需要面对的重要课题。",
            "兵马俑的考古工作自1974年被发现以来，一直是全球考古学界的焦点。考古学家通过对一号坑、二号坑和三号坑的系统发掘，逐步揭示了秦代军事装备、军队编制以及陶俑制作工艺的细节。研究发现，兵马俑的排列并非随意，而是严格按照秦代军队的实战阵型布置，前锋、中军和后卫分工明确。陶俑的制作采用了模块化生产方式，不同身体部位分别烧制后组装，体现了秦代高效的组织能力。此外，兵马俑的彩绘技术尤为精湛。考古人员发现，陶俑原本通体施彩，颜料以矿物质为主，历经两千余年仍部分保留鲜艳色泽。近年来，中德合作团队利用3D扫描和光谱分析技术，成功复原了部分陶俑的原始色彩，为研究秦代服饰文化提供了重要依据。由于氧化和湿度影响，兵马俑的保护面临巨大挑战。目前，文物保护专家采用纳米材料加固、控温控湿等先进技术延缓陶俑的退化。同时，数字化存档和虚拟复原技术也被应用于兵马俑的长期保存与研究。这些努力不仅让这一世界文化遗产得以延续，也为其他类似文物的保护提供了借鉴。",
            "随着城市化进程加速，城市绿化已成为提升居民生活质量的关键举措。许多国际大都市通过建设垂直花园、屋顶绿地和生态走廊，有效缓解了热岛效应，并改善了空气质量。例如，新加坡的'花园城市'计划通过立法要求新建建筑必须包含绿化面积，其滨海湾花园的超级树景观更是成为全球典范。绿色基础设施不仅美化环境，还具有显著的生态效益。树木和植被能吸收二氧化碳、降低噪音，并为野生动物提供栖息地。近年来，智能灌溉系统和耐旱植物的推广进一步提高了绿化项目的可持续性。巴黎的'城市森林'计划甚至提出到2030年将50%的市区面积转化为绿地，以应对气候变化。然而，城市绿化也面临土地资源有限、维护成本高等挑战。一些城市尝试通过社区参与和公私合作模式解决这些问题，如纽约的'百万树木计划'便动员了数千名志愿者参与种植。未来，结合生物多样性和低碳技术的绿化设计将成为城市发展的核心方向。"
        ]

        for _ in range(warm_up_iterations):
            try:
                requests.post(
                    self.tei_url,
                    headers=self.headers,
                    json={
                        "query": query,
                        "documents": documents,
                        "model": "rerank_vacc",
                        "return_documents": False
                    },
                    timeout=3000
                )
            except Exception as e:
                print(f"Warm-up error: {e}")
        print("Warm-up completed")
    
    def single_rerank_request(self, query: str, documents: List[str], request_id: int) -> Tuple[float, bool]:
        """单个rerank请求函数"""
        try:
            start_time = time.perf_counter()
            response = requests.post(
                self.tei_url,
                headers=self.headers,
                json={
                    "query": query,
                    "documents": documents,
                    "model": "rerank_vacc",
                    "return_documents": False  # 为了减少响应数据量
                },
                timeout=6000  # rerank可能需要更长时间
            )
            latency = time.perf_counter() - start_time
            
            if response.status_code == 200:
                return latency, True
            return latency, False
            
        except Exception as e:
            print(f"Rerank request {request_id} error: {e}")
            return -1, False
    
    def run_concurrent_test(self, token_lengths: List[int], concurrency: int = 2, iterations_per_length: int = 5):
        """运行并发rerank测试"""
        self.warm_up()
        
        all_results = {}
        
        for target_tokens in token_lengths:
            print(f"\n{'='*50}")
            print(f"Testing rerank token length: {target_tokens}")
            print(f"{'='*50}")
            
            # 生成测试数据
            query, documents = self.generate_rerank_data(target_tokens)
            
            # 计算实际token数量
            actual_tokens = self.calculate_rerank_input_length(query, documents)
            print(f"Target tokens: {target_tokens}, Actual tokens: {actual_tokens}")
            print(f"Query length: {len(self.tokenizer.encode(query, add_special_tokens=True))} tokens")
            print(f"Documents count: {len(documents)}")
            for i, doc in enumerate(documents[:5]):  # 只显示前两个文档的信息
                doc_tokens = len(self.tokenizer.encode(doc, add_special_tokens=True))
                print(f"Document {i+1}: {doc_tokens} tokens")
            latencies = []
            success_count = 0
            
            # 使用线程池执行并发请求
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                for iteration in range(iterations_per_length):
                    futures = []
                    
                    # 提交并发请求
                    for i in range(concurrency):
                        future = executor.submit(
                            self.single_rerank_request, 
                            query, documents, 
                            iteration * concurrency + i
                        )
                        futures.append(future)
                    
                    # 等待所有请求完成
                    for future in concurrent.futures.as_completed(futures):
                        latency, success = future.result()
                        if success and latency > 0:
                            latencies.append(latency)
                            success_count += 1
                            #print(f"Request completed: {latency:.3f}s")
                        elif not success:
                            print(f"Request failed for {target_tokens} tokens")
            
            # 统计结果
            if latencies:
                stats = {
                    'target_tokens': target_tokens,
                    'actual_tokens': actual_tokens,
                    'total_requests': iterations_per_length * concurrency,
                    'success_requests': success_count,
                    'success_rate': success_count / (iterations_per_length * concurrency),
                    'avg_latency': np.mean(latencies),
                    'min_latency': np.min(latencies),
                    'max_latency': np.max(latencies),
                    'p50_latency': np.percentile(latencies, 50),
                    'p90_latency': np.percentile(latencies, 90),
                    'p95_latency': np.percentile(latencies, 95),
                    'latencies': latencies,
                    'query_length': len(self.tokenizer.encode(query, add_special_tokens=True)),
                    'documents_count': len(documents),
                    'avg_doc_length': np.mean([len(self.tokenizer.encode(doc, add_special_tokens=True)) for doc in documents])
                }
                
                all_results[target_tokens] = stats
                
                # 打印当前长度结果
                self.print_stats(stats)
            else:
                print(f"No successful requests for {target_tokens} tokens")
        
        return all_results
    
    def print_stats(self, stats: Dict):
        """打印统计结果"""
        print(f"\n--- Rerank Results for {stats['target_tokens']} tokens ---")
        print(f"Success rate: {stats['success_rate']:.2%} ({stats['success_requests']}/{stats['total_requests']})")
        print(f"Average latency: {stats['avg_latency']:.4f}s")
        print(f"Min latency: {stats['min_latency']:.4f}s")
        print(f"Max latency: {stats['max_latency']:.4f}s")
        print(f"P50 latency: {stats['p50_latency']:.4f}s")
        print(f"P90 latency: {stats['p90_latency']:.4f}s")
        print(f"P95 latency: {stats['p95_latency']:.4f}s")
        print(f"Query length: {stats['query_length']} tokens")
        print(f"Documents: {stats['documents_count']} docs, avg {stats['avg_doc_length']:.1f} tokens each")
    
    def print_summary(self, all_results: Dict):
        """打印总摘要"""
        print(f"\n{'='*80}")
        print("RERANK BENCHMARK SUMMARY REPORT")
        print(f"{'='*80}")
        
        print(f"{'Tokens':<8} {'Success':<8} {'Avg(s)':<10} {'Min(s)':<10} {'Max(s)':<10} {'P90(s)':<10} {'Docs':<6} {'AvgDoc':<8}")
        print(f"{'-'*80}")
        
        for target_tokens in sorted(all_results.keys()):
            stats = all_results[target_tokens]
            print(f"{stats['target_tokens']:<8} {stats['success_rate']:>7.1%} {stats['avg_latency']:>10.3f} "
                  f"{stats['min_latency']:>10.3f} {stats['max_latency']:>10.3f} {stats['p90_latency']:>10.3f} "
                  f"{stats['documents_count']:>6} {stats['avg_doc_length']:>8.1f}")
    
    def save_results(self, all_results: Dict, filename: str = "rerank_benchmark_results.json"):
        """保存结果到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {filename}")

def main():
    # 配置测试参数
    token_lengths = [200, 800, 1500, 3000, 8000]
    concurrency = 2
    iterations_per_length = 5  # 每个长度测试5轮（rerank较慢）
    
    # 创建测试实例
    benchmark = RerankBenchmark()
    
    # 运行测试
    print("Starting rerank benchmark test...")
    print(f"Token lengths: {token_lengths}")
    print(f"Concurrency: {concurrency}")
    print(f"Iterations per length: {iterations_per_length}")
    
    results = benchmark.run_concurrent_test(
        token_lengths=token_lengths,
        concurrency=concurrency,
        iterations_per_length=iterations_per_length
    )
    
    # 打印总结和保存结果
    benchmark.print_summary(results)
    benchmark.save_results(results)

if __name__ == "__main__":
    main()