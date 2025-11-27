import time
import requests
import concurrent.futures
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
import json

class EmbeddingBenchmark:
    def __init__(self, tei_url: str = "http://localhost:9998/v1/embeddings"):
        self.tei_url = tei_url
        self.headers = {"Content-Type": "application/json"}
        # 加载 tokenizer（确保和模型一致）
        self.tokenizer = AutoTokenizer.from_pretrained("/home/tonyguo/emb_models/bge-m3-vacc/512/tokenizer")
        # 获取模型的有效 token ID 范围（排除特殊token，确保解码后是合法文本）
        self.valid_token_ids = [
            id for id in range(self.tokenizer.vocab_size)
            if not self.tokenizer.decode([id]).strip() == ""  # 排除解码后为空的token
        ]
    
    def generate_random_text_by_token_length(self, target_tokens: int) -> Tuple[str, int]:
        """
        随机生成指定token长度的合法文本（无需手动拼接句子）
        :param target_tokens: 目标token长度
        :return: (随机文本, 实际token长度)
        """
        if target_tokens <= 0:
            return "", 0
        
        # 从有效token ID中随机选择 target_tokens 个（确保解码后是合法文本）
        random_token_ids = np.random.choice(self.valid_token_ids, size=target_tokens, replace=True)
        
        # 解码成文本（skip_special_tokens=True 忽略特殊token）
        random_text = self.tokenizer.decode(random_token_ids, skip_special_tokens=True)
        
        # 验证实际token长度（防止解码后长度不一致）
        actual_tokens = len(self.tokenizer.encode(random_text, add_special_tokens=False))
        
        # 极端情况：解码后长度为0，重新生成（概率极低）
        if actual_tokens == 0:
            return self.generate_random_text_by_token_length(target_tokens)
        
        return random_text, actual_tokens
    
    def warm_up(self, warm_up_iterations: int = 10):
        """预热TEI服务（batch=1，随机文本）"""
        print("Running warm-up...")
        warm_up_text, _ = self.generate_random_text_by_token_length(800)
        for _ in range(warm_up_iterations):
            try:
                requests.post(
                    self.tei_url,
                    headers=self.headers,
                    json={"input": [warm_up_text], "model": "emb_vacc"},  # batch=1
                    timeout=3000
                )
            except Exception as e:
                print(f"Warm-up error: {e}")
        print("Warm-up completed")
    
    def single_request(self, text: str, request_id: int) -> Tuple[int, float, bool]:
        """单个请求（batch=1），返回（实际token长度，延迟，是否成功）"""
        try:
            actual_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
            start_time = time.perf_counter()
            response = requests.post(
                self.tei_url,
                headers=self.headers,
                json={"input": [text], "model": "emb_vacc"},  # 固定batch=1
                timeout=6000
            )
            latency = time.perf_counter() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) == 1:  # 验证batch=1的返回
                    return actual_tokens, latency, True
            print(f"Request {request_id} failed: status_code={response.status_code if 'response' in locals() else 'N/A'}")
            return actual_tokens, latency, False
            
        except Exception as e:
            print(f"Request {request_id} error: {e}")
            actual_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
            return actual_tokens, -1, False
    
    def run_concurrent_test(self, token_lengths: List[int], concurrency: int = 2, iterations_per_length: int = 10):
        """运行并发测试（固定batch=1，随机文本）"""
        self.warm_up()
        all_results: Dict[int, List[float]] = {}  # key: 实际token长度, value: 延迟列表
        all_request_info: List[Dict] = []

        for target_tokens in token_lengths:
            print(f"\n{'='*50}")
            print(f"Testing target token length: {target_tokens} (batch=1, random text)")
            print(f"{'='*50}")
            
            # 生成随机测试文本（每个长度生成1个文本，所有请求复用，避免文本差异影响结果）
            test_text, actual_tokens = self.generate_random_text_by_token_length(target_tokens)
            print(f"Target tokens: {target_tokens}, Actual tokens: {actual_tokens}")
            
            # 并发请求
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = []
                # 提交 iterations_per_length * concurrency 个请求
                for i in range(iterations_per_length):
                    for j in range(concurrency):
                        request_id = i * concurrency + j
                        future = executor.submit(self.single_request, test_text, request_id)
                        futures.append(future)
                
                # 收集结果
                for future in concurrent.futures.as_completed(futures):
                    req_actual_tokens, latency, success = future.result()
                    if req_actual_tokens not in all_results:
                        all_results[req_actual_tokens] = []
                    if success and latency > 0:
                        all_results[req_actual_tokens].append(latency)
                        all_request_info.append({
                            "target_tokens": target_tokens,
                            "actual_tokens": req_actual_tokens,
                            "latency": latency,
                            "success": True
                        })
                    else:
                        all_request_info.append({
                            "target_tokens": target_tokens,
                            "actual_tokens": req_actual_tokens,
                            "latency": latency,
                            "success": False
                        })
        
        # 汇总统计结果
        summarized_results = self.summarize_results(all_results, all_request_info)
        return summarized_results
    
    def summarize_results(self, all_results: Dict[int, List[float]], all_request_info: List[Dict]) -> Dict[int, Dict]:
        """汇总结果（按实际token长度分组）"""
        summarized = {}
        for actual_tokens, latencies in sorted(all_results.items()):
            reqs = [r for r in all_request_info if r["actual_tokens"] == actual_tokens]
            total = len(reqs)
            success = len([r for r in reqs if r["success"]])
            
            summarized[actual_tokens] = {
                'actual_tokens': actual_tokens,
                'target_tokens_list': sorted(list(set([r["target_tokens"] for r in reqs]))),
                'total_requests': total,
                'success_requests': success,
                'success_rate': success / total if total > 0 else 0.0,
                'avg_latency': np.mean(latencies),
                'min_latency': np.min(latencies),
                'max_latency': np.max(latencies),
                'p50_latency': np.percentile(latencies, 50),
                'p90_latency': np.percentile(latencies, 90),
                'p95_latency': np.percentile(latencies, 95),
                'latencies': latencies
            }
        return summarized
    
    def print_stats(self, stats: Dict):
        """打印单组统计结果"""
        print(f"\n--- Results for actual tokens: {stats['actual_tokens']} ---")
        print(f"Corresponding target tokens: {stats['target_tokens_list']}")
        print(f"Success rate: {stats['success_rate']:.2%} ({stats['success_requests']}/{stats['total_requests']})")
        print(f"Average latency: {stats['avg_latency']:.4f}s")
        print(f"Min latency: {stats['min_latency']:.4f}s")
        print(f"Max latency: {stats['max_latency']:.4f}s")
        print(f"P50 latency: {stats['p50_latency']:.4f}s")
        print(f"P90 latency: {stats['p90_latency']:.4f}s")
        print(f"P95 latency: {stats['p95_latency']:.4f}s")
    
    def print_summary(self, all_results: Dict):
        """打印总摘要"""
        print(f"\n{'='*80}")
        print("SUMMARY REPORT (Random Text)")
        print(f"{'='*80}")
        
        print(f"{'Actual Tokens':<12} {'Success':<8} {'Avg(s)':<8} {'Min(s)':<8} {'Max(s)':<8} {'P90(s)':<8} {'Total Reqs':<8}")
        print(f"{'-'*80}")
        
        for actual_tokens in sorted(all_results.keys()):
            stats = all_results[actual_tokens]
            print(f"{actual_tokens:<12} {stats['success_rate']:>7.1%} {stats['avg_latency']:>8.3f} "
                  f"{stats['min_latency']:>8.3f} {stats['max_latency']:>8.3f} {stats['p90_latency']:>8.3f} {stats['total_requests']:>8}")
    
    def save_results(self, all_results: Dict, filename: str = "embedding_benchmark_results.json"):
        """保存结果到文件"""
        # 转换numpy类型为json可序列化
        for stats in all_results.values():
            stats['latencies'] = [float(l) for l in stats['latencies']]
            stats['avg_latency'] = float(stats['avg_latency'])
            stats['min_latency'] = float(stats['min_latency'])
            stats['max_latency'] = float(stats['max_latency'])
            stats['p50_latency'] = float(stats['p50_latency'])
            stats['p90_latency'] = float(stats['p90_latency'])
            stats['p95_latency'] = float(stats['p95_latency'])
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {filename}")

def main():
    token_lengths = [200, 800, 1500, 3000, 8000]  # 要测试的token长度列表
    concurrency = 2  # 并发数（同时发起多少个请求）
    iterations_per_length = 10  # 每个长度测试10轮
    
    # 创建测试实例
    benchmark = EmbeddingBenchmark()
    
    # 运行测试
    print("Starting embedding benchmark test (batch=1, random text)...")
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