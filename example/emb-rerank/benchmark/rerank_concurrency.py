import time
import requests
import concurrent.futures
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
import json

class RerankBenchmark:
    def __init__(self, tei_url: str = "http://localhost:9999/v1/rerank"):
        self.tei_url = tei_url
        self.headers = {"Content-Type": "application/json"}
        self.tokenizer = AutoTokenizer.from_pretrained("/home/tonyguo/emb_models/bge-reranker-v2-m3-vacc/512/tokenizer")
        self.valid_token_ids = [
            id for id in range(self.tokenizer.vocab_size)
            if not self.tokenizer.decode([id]).strip() == ""
        ]
        self.results = {}
    
    def generate_single_pair_components(self, target_pair_tokens: int) -> Tuple[str, str, int]:
        """
        生成单个 (query+doc) pair 的组件（严格匹配 pair 编码逻辑）
        :param target_pair_tokens: 单个 pair 的目标 token 数（含 [CLS] 和 [SEP]）
        :return: (query, doc, 实际 pair token 数)
        """
        # 每个 pair 固定含 2 个特殊 token ([CLS] + [SEP])，所以普通 token 数 = 目标 - 2
        target_normal_tokens = max(50, target_pair_tokens - 2)
        
        # 分配 query 和 doc 的普通 token 比例（2:8，符合 rerank 常见场景）
        query_normal_tokens = int(target_normal_tokens * 0.2)
        doc_normal_tokens = target_normal_tokens - query_normal_tokens
        
        # 生成 query（确保普通 token 数精准）
        query_token_ids = np.random.choice(self.valid_token_ids, size=query_normal_tokens, replace=True)
        query = self.tokenizer.decode(query_token_ids, skip_special_tokens=True)
        query_actual_normal = len(self.tokenizer.encode(query, add_special_tokens=False))
        
        # 生成 doc（确保普通 token 数精准）
        doc_token_ids = np.random.choice(self.valid_token_ids, size=doc_normal_tokens, replace=True)
        doc = self.tokenizer.decode(doc_token_ids, skip_special_tokens=True)
        doc_actual_normal = len(self.tokenizer.encode(doc, add_special_tokens=False))
        
        # 按 rerank 实际编码逻辑计算 pair 长度（严格对齐 calculate_rerank_input_length）
        combined_text = f"{query} {doc}"
        pair_actual_tokens = len(self.tokenizer.encode(combined_text, add_special_tokens=True))
        
        # 微调：如果偏差超过 3%，重新生成（确保精准）
        max_attempts = 2
        attempts = 0
        while abs(pair_actual_tokens - target_pair_tokens) / target_pair_tokens > 0.03 and attempts < max_attempts:
            query_token_ids = np.random.choice(self.valid_token_ids, size=query_normal_tokens, replace=True)
            query = self.tokenizer.decode(query_token_ids, skip_special_tokens=True)
            query_actual_normal = len(self.tokenizer.encode(query, add_special_tokens=False))
            
            doc_token_ids = np.random.choice(self.valid_token_ids, size=doc_normal_tokens, replace=True)
            doc = self.tokenizer.decode(doc_token_ids, skip_special_tokens=True)
            doc_actual_normal = len(self.tokenizer.encode(doc, add_special_tokens=False))
            
            combined_text = f"{query} {doc}"
            pair_actual_tokens = len(self.tokenizer.encode(combined_text, add_special_tokens=True))
            attempts += 1
        
        return query, doc, pair_actual_tokens
    
    def calculate_rerank_input_length(self, query: str, documents: List[str]) -> int:
        """
        你的原始计算逻辑（保持不变，确保生成和计算完全对齐）
        计算 rerank 任务的总 token 数量：每个 (query+doc) pair 编码后的长度之和
        """
        total_tokens = 0
        for doc in documents:
            combined_text = f"{query} {doc}"
            tokens = self.tokenizer.encode(combined_text, add_special_tokens=True)
            total_tokens += len(tokens)
        return total_tokens
    
    def generate_rerank_data(
        self, 
        target_total_tokens: int,  # 目标总 token 数（所有 pair 之和）
        top_k: int = 5  # pair 数量（query + top_k docs）
    ) -> Tuple[str, List[str], int]:
        """
        生成 rerank 测试数据（严格对齐你的计算逻辑，总 token 数精准匹配目标）
        """
        # 1. 计算每个 pair 的目标 token 数（平均分配）
        target_pair_tokens = max(50, target_total_tokens // top_k)
        
        # 2. 生成第一个 pair，确定 query（所有 pair 共用同一个 query，符合 rerank 实际场景）
        query, first_doc, first_pair_actual = self.generate_single_pair_components(target_pair_tokens)
        
        # 3. 生成剩余 top_k-1 个 doc（共用同一个 query）
        documents = [first_doc]
        total_actual_tokens = first_pair_actual
        
        for _ in range(top_k - 1):
            # 复用 query，只生成 doc
            _, doc, pair_actual = self.generate_single_pair_components(target_pair_tokens)
            documents.append(doc)
            total_actual_tokens += pair_actual
        
        # 4. 最终校准：如果总长度和目标偏差超过 5%，微调最后一个 doc
        if abs(total_actual_tokens - target_total_tokens) / target_total_tokens > 0.05:
            # 计算需要的补偿长度
            needed_compensation = target_total_tokens - total_actual_tokens
            new_target_pair = target_pair_tokens + needed_compensation
            if new_target_pair >= 50:  # 避免 pair 过短
                _, new_doc, new_pair_actual = self.generate_single_pair_components(new_target_pair)
                documents[-1] = new_doc
                total_actual_tokens = total_actual_tokens - pair_actual + new_pair_actual
        
        # 5. 最终验证（确保和你的计算逻辑完全一致）
        final_total_tokens = self.calculate_rerank_input_length(query, documents)
        return query, documents, final_total_tokens
    
    def warm_up(self, warm_up_iterations: int = 3, top_k: int = 5):
        """预热 Rerank 服务"""
        print("Running warm-up...")
        query, documents, _ = self.generate_rerank_data(target_total_tokens=300, top_k=top_k)
        
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
    
    def single_rerank_request(self, query: str, documents: List[str], request_id: int) -> Tuple[int, float, bool]:
        """单个 rerank 请求函数"""
        try:
            total_actual_tokens = self.calculate_rerank_input_length(query, documents)
            start_time = time.perf_counter()
            response = requests.post(
                self.tei_url,
                headers=self.headers,
                json={
                    "query": query,
                    "documents": documents,
                    "model": "rerank_vacc",
                    "return_documents": False
                },
                timeout=6000
            )
            latency = time.perf_counter() - start_time
            
            if response.status_code == 200:
                return total_actual_tokens, latency, True
            print(f"Request {request_id} failed: status_code={response.status_code}")
            return total_actual_tokens, latency, False
            
        except Exception as e:
            print(f"Rerank request {request_id} error: {e}")
            total_actual_tokens = self.calculate_rerank_input_length(query, documents)
            return total_actual_tokens, -1, False
    
    def run_concurrent_test(
        self, 
        target_total_tokens_list: List[int],  # 目标总 token 数列表（核心配置）
        top_k_list: List[int] = [5],  # pair 数量列表
        concurrency: int = 2, 
        iterations_per_combination: int = 5
    ):
        """运行并发 rerank 测试"""
        self.warm_up(top_k=max(top_k_list))
        all_results: Dict[Tuple[int, int], List[float]] = {}  # key: (实际总 token 数, top-k)
        all_request_info: List[Dict] = []

        for target_total in target_total_tokens_list:
            for top_k in top_k_list:
                print(f"\n{'='*60}")
                print(f"Testing: Target Total Tokens = {target_total}, Top-k = {top_k}")
                print(f"{'='*60}")
                
                # 生成测试数据（严格对齐计算逻辑）
                query, documents, actual_total = self.generate_rerank_data(
                    target_total_tokens=target_total,
                    top_k=top_k
                )
                
                # 打印详细信息（验证每个 pair 长度和总长度）
                pair_lengths = []
                for doc in documents:
                    combined_text = f"{query} {doc}"
                    pair_len = len(self.tokenizer.encode(combined_text, add_special_tokens=True))
                    pair_lengths.append(pair_len)
                
                print(f"Target Total Tokens: {target_total}, Actual Total: {actual_total} (偏差: {abs(actual_total-target_total)/target_total:.1%})")
                print(f"Pair Count: {top_k}")
                print(f"Each Pair Lengths: {[int(x) for x in pair_lengths]}")
                print(f"Avg Pair Length: {np.mean(pair_lengths):.0f} tokens")
                print(f"Query Length: {len(self.tokenizer.encode(query, add_special_tokens=False))} tokens (普通 token)")
                
                # 并发请求
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = []
                    for i in range(iterations_per_combination):
                        for j in range(concurrency):
                            request_id = i * concurrency + j
                            future = executor.submit(
                                self.single_rerank_request,
                                query, documents, request_id
                            )
                            futures.append(future)
                    
                    # 收集结果
                    for future in concurrent.futures.as_completed(futures):
                        total_tokens, latency, success = future.result()
                        key = (int(round(total_tokens, -2)), top_k)  # 按总 token 数四舍五入分组
                        if key not in all_results:
                            all_results[key] = []
                        
                        if success and latency > 0:
                            all_results[key].append(latency)
                            all_request_info.append({
                                "target_total_tokens": target_total,
                                "actual_total_tokens": total_tokens,
                                "top_k": top_k,
                                "latency": latency,
                                "success": True
                            })
                        else:
                            all_request_info.append({
                                "target_total_tokens": target_total,
                                "actual_total_tokens": total_tokens,
                                "top_k": top_k,
                                "latency": latency,
                                "success": False
                            })
                            print(f"Request failed: Target={target_total}, Top-k={top_k}")
        
        # 汇总统计结果
        summarized_results = self.summarize_results(all_results, all_request_info)
        return summarized_results
    
    def summarize_results(self, all_results: Dict[Tuple[int, int], List[float]], all_request_info: List[Dict]) -> Dict[str, Dict]:
        """汇总结果"""
        summarized = {}
        for (actual_total_tokens, top_k), latencies in sorted(all_results.items()):
            reqs = [r for r in all_request_info if 
                    int(round(r["actual_total_tokens"], -2)) == actual_total_tokens and 
                    r["top_k"] == top_k]
            total = len(reqs)
            success = len([r for r in reqs if r["success"]])
            
            if latencies:
                key = f"{actual_total_tokens}-top{top_k}"
                summarized[key] = {
                    'target_total_tokens_list': sorted(list(set([r["target_total_tokens"] for r in reqs]))),
                    'actual_avg_total_tokens': int(np.mean([r["actual_total_tokens"] for r in reqs])),
                    'top_k': top_k,
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
        print(f"\n--- Results: Actual Avg Total Tokens = {stats['actual_avg_total_tokens']}, Top-k = {stats['top_k']} ---")
        print(f"Corresponding Targets: {stats['target_total_tokens_list']}")
        print(f"Success rate: {stats['success_rate']:.2%} ({stats['success_requests']}/{stats['total_requests']})")
        print(f"Average latency: {stats['avg_latency']:.4f}s")
        print(f"Min latency: {stats['min_latency']:.4f}s")
        print(f"Max latency: {stats['max_latency']:.4f}s")
        print(f"P50 latency: {stats['p50_latency']:.4f}s")
        print(f"P90 latency: {stats['p90_latency']:.4f}s")
        print(f"P95 latency: {stats['p95_latency']:.4f}s")
    
    def print_summary(self, all_results: Dict):
        """打印总摘要"""
        print(f"\n{'='*100}")
        print("RERANK BENCHMARK SUMMARY REPORT (Aligned with Your Calculation Logic)")
        print(f"{'='*100}")
        
        print(f"{'Actual Avg Total':<18} {'Top-k':<6} {'Success':<8} {'Avg(s)':<10} {'Min(s)':<10} {'Max(s)':<10} {'P90(s)':<10} {'Total Reqs':<8}")
        print(f"{'-'*100}")
        
        sorted_keys = sorted(all_results.keys(), key=lambda x: (int(x.split('-')[0]), int(x.split('top')[1])))
        for key in sorted_keys:
            stats = all_results[key]
            print(f"{stats['actual_avg_total_tokens']:<18} {stats['top_k']:<6} {stats['success_rate']:>7.1%} {stats['avg_latency']:>10.3f} "
                  f"{stats['min_latency']:>10.3f} {stats['max_latency']:>10.3f} {stats['p90_latency']:>10.3f} {stats['total_requests']:<8}")
    
    def save_results(self, all_results: Dict, filename: str = "rerank_benchmark_results.json"):
        """保存结果到文件"""
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
    # 配置测试参数（直接填写目标总 token 数）
    target_total_tokens_list = [300, 1000, 2000, 3000, 8000]
    top_k_list = [5]  # 可扩展为 [3,5,10]
    concurrency = 2
    iterations_per_combination = 5
    
    benchmark = RerankBenchmark()
    
    print("Starting rerank benchmark test (Strictly Aligned with Your Calculation Logic)...")
    print(f"Target Total Tokens List: {target_total_tokens_list}")
    print(f"Top-k List: {top_k_list}")
    print(f"Concurrency: {concurrency}")
    print(f"Iterations per Combination: {iterations_per_combination}")
    
    results = benchmark.run_concurrent_test(
        target_total_tokens_list=target_total_tokens_list,
        top_k_list=top_k_list,
        concurrency=concurrency,
        iterations_per_combination=iterations_per_combination
    )
    
    benchmark.print_summary(results)
    benchmark.save_results(results)

if __name__ == "__main__":
    main()