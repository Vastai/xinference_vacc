import time
import requests
import concurrent.futures
import threading
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
import json

class EmbeddingBenchmark:
    def __init__(self, tei_url: str = "http://localhost:9998/v1/embeddings"):
        self.tei_url = tei_url
        self.headers = {"Content-Type": "application/json"}
        self.tokenizer = AutoTokenizer.from_pretrained("/home/tonyguo/emb_models/bge-m3-vacc/tokenizer")
        self.latency_lock = threading.Lock()
        self.results = {}
    
    def generate_text_by_token_length(self, target_tokens: int, base_text: str = None) -> str:
        """生成指定token长度的文本（使用预定义句子循环拼接，确保不超target，语义完整）"""
        sentences_short = [
            "人工智能与人类的关系正在经历从简单工具到深度协作的转变，这种变革已经渗透到各个行业的核心环节。",
            "在医疗诊断领域，最新一代的AI辅助系统不仅能够以超过96%的准确率识别医学影像中的微小病灶，更能结合患者的电子病历和基因组数据给出个性化治疗建议。",
            "然而真正具有突破性的是人机协作诊断模式——AI负责处理海量数据，医生则专注于与患者的深度沟通，将冰冷的检测结果转化为温暖的治疗方案。",
            "教育行业正在经历类似的变革，智能教学系统可以实时追踪每个学生的知识掌握曲线，自动生成千人千面的练习题。",
            "而教师角色则转向更高级的育人工作，他们利用AI节省出的时间组织项目式学习，带领学生开展跨学科课题研究。",
            "在北京某实验中学的实践中，这种模式使得学生的批判性思维得分提升了37%，而基础知识掌握率反而提高了15%。",
            "制造业的人机协作展现出更丰富的维度，在特斯拉的超级工厂里，AI视觉检测系统每秒钟可以完成200个零部件的质量筛查。",
            "但真正提升良品率的关键在于经验丰富的工程师，他们能够从AI检测出的异常模式中发现生产设备的潜在故障。",
            "这种人类经验与机器效率的结合，使得某条产线的故障预警时间从原来的72小时缩短到惊人的45分钟。",
            "艺术创作领域见证了最令人惊叹的协作形式，最新的人工智能作曲工具可以生成复杂的交响乐章。",
            "但真正打动听众的作品往往来自人类音乐家的二次创作，他们会调整AI生成的旋律线条，注入真实的情感体验。",
            "2024年格莱美奖最佳电子音乐专辑就是由音乐人与AI系统共同署名，这标志着艺术界对人机协作的正式认可。",
            "这种深度协作揭示了一个本质规律：人工智能最适合处理确定性的计算任务，而人类则专注于需要价值判断和情感共鸣的领域。",
            "未来社会需要的不是与AI竞争的工作者，而是懂得如何与AI协作的新型人才。",
            "教育体系必须做出相应变革，重点培养机器难以替代的创造力、同理心和系统思维能力，这将是人机协同时代最重要的核心竞争力。"
        ]

        if target_tokens <= 0:
            return ""

        result_parts = []
        current_token_count = 0
        sentence_index = 0

        while True:
            sentence = sentences_short[sentence_index % len(sentences_short)]
            # 尝试添加当前句子（前面加空格分隔）
            candidate = " ".join(result_parts + [sentence]) if result_parts else sentence
            tokens = self.tokenizer.encode(candidate, add_special_tokens=False)
            new_length = len(tokens)

            # 如果加上这句会超，就停止（保证不超限）
            if new_length > target_tokens:
                break

            # 否则接受这个句子
            result_parts.append(sentence)
            current_token_count = new_length
            sentence_index += 1

            # 防止无限循环（极端小 target）
            if sentence_index > 1000:
                break

        final_text = " ".join(result_parts)
        
        # 可选：输出实际长度（调试用）
        # actual = len(self.tokenizer.encode(final_text, add_special_tokens=False))
        # print(f"[INFO] Target: {target_tokens}, Generated: {actual} tokens")

        return final_text
    
    def warm_up(self, warm_up_iterations: int = 10):
        """预热TEI服务"""
        print("Running warm-up...")
        warm_up_texts = [
        "人工智能与人类的关系正在经历从简单工具到深度协作的转变，这种变革已经渗透到各个行业的核心环节。"
        "在医疗诊断领域，最新一代的AI辅助系统不仅能够以超过96%的准确率识别医学影像中的微小病灶，更能结合患者的电子病历和基因组数据给出个性化治疗建议。"
        "然而真正具有突破性的是人机协作诊断模式——AI负责处理海量数据，医生则专注于与患者的深度沟通，将冰冷的检测结果转化为温暖的治疗方案。",
        ] * 10
        for _ in range(warm_up_iterations):
            try:
                requests.post(
                    self.tei_url,
                    headers=self.headers,
                    json={"input": warm_up_texts, "model": "emb_vacc"},
                    timeout=3000
                )
            except Exception as e:
                print(f"Warm-up error: {e}")
        print("Warm-up completed")
    
    def single_request(self, sentences: List[str], request_id: int) -> Tuple[float, bool]:
        """单个请求函数"""
        try:
            start_time = time.perf_counter()
            response = requests.post(
                self.tei_url,
                headers=self.headers,
                json={"input": sentences, "model": "emb_vacc"},
                timeout=6000
            )
            latency = time.perf_counter() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) > 0:
                    return latency, True
            return latency, False
            
        except Exception as e:
            print(f"Request {request_id} error: {e}")
            return -1, False
    
    def run_concurrent_test(self, token_lengths: List[int], concurrency: int = 2, iterations_per_length: int = 10):
        """运行并发测试"""
        self.warm_up()
        all_results = {}
        
        for target_tokens in token_lengths:
            print(f"\n{'='*50}")
            print(f"Testing token length: {target_tokens}")
            print(f"{'='*50}")
            
            # 生成测试文本
            test_text = self.generate_text_by_token_length(target_tokens)
            actual_tokens = len(self.tokenizer.encode(test_text, add_special_tokens=False))
            print(f"Target tokens: {target_tokens}, Actual tokens: {actual_tokens}")
            
            sentences = [test_text]
            latencies = []
            success_count = 0
            
            # 使用线程池执行并发请求
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                for iteration in range(iterations_per_length):
                    futures = []
                    
                    # 提交并发请求
                    for i in range(concurrency):
                        future = executor.submit(self.single_request, sentences, iteration * concurrency + i)
                        futures.append(future)
                    
                    # 等待所有请求完成
                    for future in concurrent.futures.as_completed(futures):
                        latency, success = future.result()
                        if success and latency > 0:
                            latencies.append(latency)
                            success_count += 1
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
                    'latencies': latencies
                }
                
                all_results[target_tokens] = stats
                
                # 打印当前长度结果
                self.print_stats(stats)
            else:
                print(f"No successful requests for {target_tokens} tokens")
        
        return all_results
    
    def print_stats(self, stats: Dict):
        """打印统计结果"""
        print(f"\n--- Results for {stats['target_tokens']} tokens ---")
        print(f"Success rate: {stats['success_rate']:.2%} ({stats['success_requests']}/{stats['total_requests']})")
        print(f"Average latency: {stats['avg_latency']:.4f}s")
        print(f"Min latency: {stats['min_latency']:.4f}s")
        print(f"Max latency: {stats['max_latency']:.4f}s")
        print(f"P50 latency: {stats['p50_latency']:.4f}s")
        print(f"P90 latency: {stats['p90_latency']:.4f}s")
        print(f"P95 latency: {stats['p95_latency']:.4f}s")
    
    def print_summary(self, all_results: Dict):
        """打印总摘要"""
        print(f"\n{'='*60}")
        print("SUMMARY REPORT")
        print(f"{'='*60}")
        
        print(f"{'Tokens':<8} {'Success':<8} {'Avg(s)':<8} {'Min(s)':<8} {'Max(s)':<8} {'P90(s)':<8}")
        print(f"{'-'*60}")
        
        for target_tokens in sorted(all_results.keys()):
            stats = all_results[target_tokens]
            print(f"{stats['target_tokens']:<8} {stats['success_rate']:>7.1%} {stats['avg_latency']:>8.3f} "
                  f"{stats['min_latency']:>8.3f} {stats['max_latency']:>8.3f} {stats['p90_latency']:>8.3f}")
    
    def save_results(self, all_results: Dict, filename: str = "embedding_benchmark_results.json"):
        """保存结果到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {filename}")

def main():
    # 配置测试参数
    token_lengths = [200, 800, 1500, 3000, 8000]
    concurrency = 2
    iterations_per_length = 10  # 每个长度测试10轮
    
    # 创建测试实例
    benchmark = EmbeddingBenchmark()
    
    # 运行测试
    print("Starting embedding benchmark test...")
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