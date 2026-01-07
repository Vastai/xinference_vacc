import time
import requests
import concurrent.futures
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
import json
import csv 

class EmbeddingBenchmark:
    def __init__(self, tei_url: str = "http://localhost:9997/v1/embeddings", model_name: str = "Qwen3-Embedding-0.6B"):
        self.tei_url = tei_url
        self.model_name = model_name
        self.headers = {"Content-Type": "application/json"}
        # 加载 tokenizer（确保和模型一致）
        self.tokenizer = AutoTokenizer.from_pretrained(f"/FS03/weights/{self.model_name}")
        # 定义基础文本（用于拼接成固定token长度，选择简单词汇避免解码异常）
        self.base_text = "测试文本 "  # 基础短句，确保token化后长度稳定
    
    def generate_fixed_length_text(self, target_tokens: int) -> Tuple[str, int]:
        """
        生成固定token长度的文本（非随机，精准匹配目标长度）
        :param target_tokens: 目标token长度
        :return: (固定文本, 实际token长度)
        """
        if target_tokens <= 0:
            return "", 0
        
        # 步骤1：获取基础文本的token长度
        base_tokens = len(self.tokenizer.encode(self.base_text, add_special_tokens=False))
        if base_tokens == 0:
            self.base_text = "a "  # 兜底：确保基础文本有token
            base_tokens = 1
        
        # 步骤2：计算需要重复的次数，凑出目标长度
        repeat_times = target_tokens // base_tokens
        remainder = target_tokens % base_tokens
        
        # 步骤3：拼接基础文本 + 补充剩余token
        fixed_text = self.base_text * repeat_times
        if remainder > 0:
            # 从基础文本中截取前remainder个token的内容
            base_token_ids = self.tokenizer.encode(self.base_text, add_special_tokens=False)[:remainder]
            fixed_text += self.tokenizer.decode(base_token_ids, skip_special_tokens=True)
        
        # 验证实际token长度（确保精准匹配）
        actual_tokens = len(self.tokenizer.encode(fixed_text, add_special_tokens=False))
        
        # 极端情况：长度不匹配，微调（概率极低）
        if actual_tokens != target_tokens:
            if actual_tokens > target_tokens:
                # 截取前target_tokens个token
                token_ids = self.tokenizer.encode(fixed_text, add_special_tokens=False)[:target_tokens]
                fixed_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                actual_tokens = target_tokens
            else:
                # 补充基础文本直到长度匹配
                while actual_tokens < target_tokens:
                    fixed_text += self.base_text[:1]  # 每次加1个字符
                    actual_tokens = len(self.tokenizer.encode(fixed_text, add_special_tokens=False))
                    if actual_tokens > target_tokens:
                        token_ids = self.tokenizer.encode(fixed_text, add_special_tokens=False)[:target_tokens]
                        fixed_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                        actual_tokens = target_tokens
        
        return fixed_text, actual_tokens
    
    def warm_up(self, warm_up_iterations: int = 3, warm_up_token_length: int = 800):
        """预热TEI服务（batch=1，固定长度文本）"""
        print(f"Running warm-up for model: {self.model_name}...")
        warm_up_text, _ = self.generate_fixed_length_text(warm_up_token_length)
        for _ in range(warm_up_iterations):
            try:
                requests.post(
                    self.tei_url,
                    headers=self.headers,
                    json={"input": [warm_up_text], "model": self.model_name},
                    timeout=30000000
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
                json={"input": [text], "model": self.model_name},
                timeout=60000000
            )
            latency = time.perf_counter() - start_time
            
            if response.status_code == 200:
                data = response.json()
                if "data" in data and len(data["data"]) == 1:
                    return actual_tokens, latency, True
            print(f"Request {request_id} failed: status_code={response.status_code if 'response' in locals() else 'N/A'}")
            return actual_tokens, latency, False
            
        except Exception as e:
            print(f"Request {request_id} error: {e}")
            actual_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
            return actual_tokens, -1, False
    
    def run_concurrent_test(
        self, 
        concurrency_token_mapping: Dict[int, List[int]],  # 核心：并发数->token长度列表的映射
        iterations_per_length: int = 10
    ):
        """
        运行并发测试（固定batch=1，固定长度文本）
        :param concurrency_token_mapping: 例如 {1: [5,10], 2: [5,20], 4: [5]}
        """
        # 预热使用最大token长度
        all_token_lengths = []
        for lengths in concurrency_token_mapping.values():
            all_token_lengths.extend(lengths)
        max_token = max(all_token_lengths) if all_token_lengths else 800
        self.warm_up(warm_up_token_length=max_token)
        
        all_results: Dict[Tuple[int, int], List[float]] = {}  # key: (实际token长度, 并发数)
        all_request_info: List[Dict] = []

        # 遍历每个并发数及其对应的token长度列表
        for concurrency, token_lengths in concurrency_token_mapping.items():
            print(f"\n{'='*60}")
            print(f"Starting tests for Concurrency: {concurrency}")
            print(f"Token lengths for this concurrency: {token_lengths}")
            print(f"{'='*60}")
            
            for target_tokens in token_lengths:
                print(f"\n{'='*50}")
                print(f"Testing target token length: {target_tokens} (batch=1, fixed text), Concurrency: {concurrency}")
                print(f"{'='*50}")
                
                # 生成固定长度文本（精准匹配target_tokens）
                test_text, actual_tokens = self.generate_fixed_length_text(target_tokens)
                print(f"Target tokens: {target_tokens}, Actual tokens: {actual_tokens}, Concurrency: {concurrency}")
                print(f"Sample text: {test_text[:50]}..." if len(test_text) > 50 else f"Sample text: {test_text}")
                
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
                        key = (req_actual_tokens, concurrency)
                        if key not in all_results:
                            all_results[key] = []
                        if success and latency > 0:
                            all_results[key].append(latency)
                            all_request_info.append({
                                "target_tokens": target_tokens,
                                "actual_tokens": req_actual_tokens,
                                "latency": latency,
                                "success": True,
                                "concurrency": concurrency
                            })
                        else:
                            all_request_info.append({
                                "target_tokens": target_tokens,
                                "actual_tokens": req_actual_tokens,
                                "latency": latency,
                                "success": False,
                                "concurrency": concurrency
                            })
        
        # 汇总统计结果
        summarized_results = self.summarize_results(all_results, all_request_info)
        return summarized_results
    
    def summarize_results(self, all_results: Dict[Tuple[int, int], List[float]], all_request_info: List[Dict]) -> Dict[str, Dict]:
        """汇总结果（按实际token长度+并发数分组）"""
        summarized = {}
        for (actual_tokens, concurrency), latencies in sorted(all_results.items()):
            reqs = [r for r in all_request_info if r["actual_tokens"] == actual_tokens and r["concurrency"] == concurrency]
            total = len(reqs)
            success = len([r for r in reqs if r["success"]])
            
            key = f"{actual_tokens}-conc{concurrency}"  # 构建包含并发数的key
            summarized[key] = {
                'actual_tokens': actual_tokens,
                'concurrency': concurrency,
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
                'latencies': latencies,
                'throughput': concurrency * actual_tokens/np.mean(latencies)
            }
        return summarized
    
    def print_stats(self, stats: Dict):
        """打印单组统计结果"""
        print(f"\n--- Results for actual tokens: {stats['actual_tokens']}, Concurrency: {stats['concurrency']} ---")
        print(f"Corresponding target tokens: {stats['target_tokens_list']}")
        print(f"Success rate: {stats['success_rate']:.2%} ({stats['success_requests']}/{stats['total_requests']})")
        print(f"Average latency: {stats['avg_latency']:.4f}s")
        print(f"Min latency: {stats['min_latency']:.4f}s")
        print(f"Max latency: {stats['max_latency']:.4f}s")
        print(f"P50 latency: {stats['p50_latency']:.4f}s")
        print(f"P90 latency: {stats['p90_latency']:.4f}s")
        print(f"P95 latency: {stats['p95_latency']:.4f}s")
        print(f"Throughput (tokens/s): {stats['throughput']:.2f}")
    
    def print_summary(self, all_results: Dict):
        """打印总摘要"""
        print(f"\n{'='*100}")
        print(f"SUMMARY REPORT (Model: {self.model_name}, Fixed Text)")
        print(f"{'='*100}")
        
        print(f"{'Actual Tokens':<12} {'Concurrency':<10} {'Success':<8} {'Avg(s)':<8} {'Min(s)':<8} {'Max(s)':<8} {'P90(s)':<8} {'Total Reqs':<8} {'Throughput':<12}")
        print(f"{'-'*100}")
        
        # 按并发数升序、token数升序排序
        sorted_keys = sorted(all_results.keys(), key=lambda x: (int(x.split('conc')[1]), int(x.split('-')[0])))
        for key in sorted_keys:
            stats = all_results[key]
            print(f"{stats['actual_tokens']:<12} {stats['concurrency']:<10} {stats['success_rate']:>7.1%} {stats['avg_latency']:>8.3f} "
                  f"{stats['min_latency']:>8.3f} {stats['max_latency']:>8.3f} {stats['p90_latency']:>8.3f} {stats['total_requests']:>8} {stats['throughput']:>12.2f}")
    
    def save_results(self, all_results: Dict, filename: str = None):
        """保存结果到文件（适配并发数）"""
        # 自动生成带模型名的文件名
        if filename is None:
            filename = f"embedding_benchmark_{self.model_name}_fixed_text_results.json"
        
        # 转换numpy类型为json可序列化
        for stats in all_results.values():
            stats['latencies'] = [float(l) for l in stats['latencies']]
            stats['avg_latency'] = float(stats['avg_latency'])
            stats['min_latency'] = float(stats['min_latency'])
            stats['max_latency'] = float(stats['max_latency'])
            stats['p50_latency'] = float(stats['p50_latency'])
            stats['p90_latency'] = float(stats['p90_latency'])
            stats['p95_latency'] = float(stats['p95_latency'])
            stats['throughput'] = float(stats['throughput'])
            stats['concurrency'] = int(stats['concurrency'])
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {filename}")

    def save_csv(self, all_results: Dict, filename: str = None):
        """保存关键指标到 CSV（按并发数升序排序）"""
        # 自动生成带模型名的文件名
        if filename is None:
            filename = f"embedding_benchmark_{self.model_name}_fixed_text_results.csv"
        
        fieldnames = [
            'actual_tokens',
            'concurrency',
            'target_tokens_list',
            'success_requests',
            'avg_latency',
            'p50_latency',
            'p90_latency',
            'p95_latency',
            'min_latency',
            'max_latency',
            'throughput'
        ]
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # 核心排序逻辑：先按并发数升序，再按实际token数升序
            sorted_keys = sorted(
                all_results.keys(),
                key=lambda x: (int(x.split('conc')[1]), int(x.split('-')[0]))
            )
            
            for key in sorted_keys:
                stats = all_results[key]
                row = {
                    'actual_tokens': stats['actual_tokens'],
                    'concurrency': stats['concurrency'],
                    'target_tokens_list': ','.join(map(str, stats['target_tokens_list'])),
                    'success_requests': stats['success_requests'],
                    'avg_latency': f"{stats['avg_latency']:.6f}",
                    'p50_latency': f"{stats['p50_latency']:.6f}",
                    'p90_latency': f"{stats['p90_latency']:.6f}",
                    'p95_latency': f"{stats['p95_latency']:.6f}",
                    'min_latency': f"{stats['min_latency']:.6f}",
                    'max_latency': f"{stats['max_latency']:.6f}",
                    'throughput': f"{stats['throughput']:.2f}"
                }
                writer.writerow(row)
        print(f"✅ CSV results saved to {filename}")

def main():
    # 核心配置：并发数 -> 固定token长度列表的映射字典（比如全测token长度5）
    concurrency_token_mapping = {
        2: [128,256,1024,4096],       # 并发2：仅测试token长度5
        4: [128,256,1024,4096],       # 并发4：仅测试token长度5
        # 如需测试更多长度，可扩展：
        # 8: [5, 10, 20]
    }
    iterations_per_length = 2  # 每个长度测试2轮（可根据需要调整）
    model_name = "Qwen3-Embedding-0.6B"      # 可切换为其他模型，如 "Qwen3-Embedding-0.6B"
    tei_url = "http://localhost:9999/v1/embeddings"
    
    # 创建测试实例
    benchmark = EmbeddingBenchmark(tei_url=tei_url, model_name=model_name)
    
    # 运行测试
    print("Starting embedding benchmark test (batch=1, fixed length text)...")
    print(f"Model name: {model_name}")
    print(f"Concurrency-Token mapping: {concurrency_token_mapping}")
    print(f"Iterations per length: {iterations_per_length}")
    
    results = benchmark.run_concurrent_test(
        concurrency_token_mapping=concurrency_token_mapping,
        iterations_per_length=iterations_per_length
    )
    
    # 打印详细统计和总结
    for key in sorted(results.keys(), key=lambda x: (int(x.split('conc')[1]), int(x.split('-')[0]))):
        benchmark.print_stats(results[key])
    
    benchmark.print_summary(results)
    benchmark.save_results(results)
    benchmark.save_csv(results)


if __name__ == "__main__":
    main()
