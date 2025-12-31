import time
import requests
import concurrent.futures
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
import json
import csv

class RerankBenchmark:
    def __init__(self, tei_url: str = "http://localhost:9998/v1/rerank", model_name: str = "bge-reranker-v2-m3"):
        self.tei_url = tei_url
        self.model_name = model_name  # 新增模型名参数，适配不同rerank模型
        self.headers = {"Content-Type": "application/json"}
        self.tokenizer = AutoTokenizer.from_pretrained(f"/disk/models/{self.model_name}")
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
        原始计算逻辑（保持不变，确保生成和计算完全对齐）
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
    
    def warm_up(self, warm_up_config: Dict = None, warm_up_iterations: int = 3):
        """预热 Rerank 服务（适配动态配置）"""
        print(f"Running warm-up for model: {self.model_name}...")
        # 默认预热配置
        if warm_up_config is None:
            warm_up_config = {"target_total_tokens": 500, "top_k": 5}
        
        query, documents, _ = self.generate_rerank_data(
            target_total_tokens=warm_up_config["target_total_tokens"],
            top_k=warm_up_config["top_k"]
        )
        
        for _ in range(warm_up_iterations):
            try:
                requests.post(
                    self.tei_url,
                    headers=self.headers,
                    json={
                        "query": query,
                        "documents": documents,
                        "model": self.model_name,  # 使用配置的模型名
                        "return_documents": False
                    },
                    timeout=300000000
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
                    "model": self.model_name,  # 使用配置的模型名
                    "return_documents": False
                },
                timeout=600000000
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
        concurrency_config_mapping: Dict[int, List[Dict]],  # 核心：并发数->测试配置列表的映射
        iterations_per_config: int = 5
    ):
        """
        运行并发 rerank 测试（支持不同并发数对应不同测试配置）
        :param concurrency_config_mapping: 示例：
            {
                1: [{"target_total_tokens": 300, "top_k": 5}, {"target_total_tokens": 600, "top_k": 5}],
                2: [{"target_total_tokens": 600, "top_k": 5}, {"target_total_tokens": 1000, "top_k": 10}],
                4: [{"target_total_tokens": 300, "top_k": 5}, {"target_total_tokens": 600, "top_k": 10}, {"target_total_tokens": 1000, "top_k": 10}]
            }
        """
        # 预热：提取所有配置中的最大值
        all_configs = []
        for configs in concurrency_config_mapping.values():
            all_configs.extend(configs)
        
        if all_configs:
            max_total_tokens = max([c["target_total_tokens"] for c in all_configs])
            max_top_k = max([c["top_k"] for c in all_configs])
            warm_up_config = {"target_total_tokens": max_total_tokens, "top_k": max_top_k}
            self.warm_up(warm_up_config=warm_up_config)
        else:
            self.warm_up()
        
        all_results: Dict[Tuple[int, int, int], List[float]] = {}  # key: (实际总 token 数, top-k, 并发数)
        all_request_info: List[Dict] = []

        # 遍历每个并发数及其对应的测试配置列表
        for concurrency, test_configs in concurrency_config_mapping.items():
            print(f"\n{'='*70}")
            print(f"Starting tests for Concurrency: {concurrency}")
            print(f"Test configs for this concurrency: {test_configs}")
            print(f"{'='*70}")
            
            for config in test_configs:
                target_total_tokens = config["target_total_tokens"]
                top_k = config["top_k"]
                
                print(f"\n{'='*60}")
                print(f"Testing: Target Total Tokens = {target_total_tokens}, Top-k = {top_k}, Concurrency = {concurrency}")
                print(f"{'='*60}")
                
                # 生成测试数据（严格对齐计算逻辑）
                query, documents, actual_total = self.generate_rerank_data(
                    target_total_tokens=target_total_tokens,
                    top_k=top_k
                )
                
                # 打印详细信息（验证每个 pair 长度和总长度）
                pair_lengths = []
                for doc in documents:
                    combined_text = f"{query} {doc}"
                    pair_len = len(self.tokenizer.encode(combined_text, add_special_tokens=True))
                    pair_lengths.append(pair_len)
                
                print(f"Target Total Tokens: {target_total_tokens}, Actual Total: {actual_total} (偏差: {abs(actual_total-target_total_tokens)/target_total_tokens:.1%})")
                print(f"Pair Count (Top-k): {top_k}")
                print(f"Each Pair Lengths: {[int(x) for x in pair_lengths]}")
                print(f"Avg Pair Length: {np.mean(pair_lengths):.0f} tokens")
                print(f"Query Length: {len(self.tokenizer.encode(query, add_special_tokens=False))} tokens (普通 token)")
                print(f"Concurrency Level: {concurrency}")
                
                # 并发请求（按当前并发数执行）
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                    futures = []
                    for i in range(iterations_per_config):
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
                        key = (int(round(total_tokens, -2)), top_k, concurrency)  
                        if key not in all_results:
                            all_results[key] = []
                        
                        if success and latency > 0:
                            all_results[key].append(latency)
                            all_request_info.append({
                                "target_total_tokens": target_total_tokens,
                                "actual_total_tokens": total_tokens,
                                "top_k": top_k,
                                "latency": latency,
                                "success": True,
                                "concurrency": concurrency
                            })
                        else:
                            all_request_info.append({
                                "target_total_tokens": target_total_tokens,
                                "actual_total_tokens": total_tokens,
                                "top_k": top_k,
                                "latency": latency,
                                "success": False,
                                "concurrency": concurrency
                            })
                            print(f"Request failed: Target={target_total_tokens}, Top-k={top_k}, Concurrency={concurrency}")
        
        # 汇总统计结果
        summarized_results = self.summarize_results(all_results, all_request_info)
        return summarized_results
    
    def summarize_results(self, all_results: Dict[Tuple[int, int, int], List[float]], all_request_info: List[Dict]) -> Dict[str, Dict]:
        """汇总结果（适配动态配置）"""
        summarized = {}
        for (actual_total_tokens, top_k, concurrency), latencies in sorted(all_results.items()):
            reqs = [r for r in all_request_info if 
                    int(round(r["actual_total_tokens"], -2)) == actual_total_tokens and 
                    r["top_k"] == top_k and
                    r["concurrency"] == concurrency]
            total = len(reqs)
            success = len([r for r in reqs if r["success"]])
            
            if latencies:
                key = f"{actual_total_tokens}-top{top_k}-conc{concurrency}"
                summarized[key] = {
                    'target_total_tokens_list': sorted(list(set([r["target_total_tokens"] for r in reqs]))),
                    'actual_avg_total_tokens': int(np.mean([r["actual_total_tokens"] for r in reqs])),
                    'top_k': top_k,
                    'concurrency': concurrency,
                    'total_requests': total,
                    'success_requests': success,  # 保留成功请求数字段
                    'success_rate': success / total if total > 0 else 0.0,
                    'avg_latency': np.mean(latencies),
                    'min_latency': np.min(latencies),
                    'max_latency': np.max(latencies),
                    'p50_latency': np.percentile(latencies, 50),
                    'p90_latency': np.percentile(latencies, 90),
                    'p95_latency': np.percentile(latencies, 95),
                    'latencies': latencies,
                    'throughput': concurrency * actual_total_tokens/np.mean(latencies)
                }
        return summarized
    
    def print_stats(self, stats: Dict):
        """打印单组统计结果"""
        print(f"\n--- Results: Actual Avg Total Tokens = {stats['actual_avg_total_tokens']}, Top-k = {stats['top_k']}, Concurrency = {stats['concurrency']} ---")
        print(f"Corresponding Targets: {stats['target_total_tokens_list']}")
        print(f"Success rate: {stats['success_rate']:.2%} ({stats['success_requests']}/{stats['total_requests']})")
        print(f"Average latency: {stats['avg_latency']:.4f}s")
        print(f"Min latency: {stats['min_latency']:.4f}s")
        print(f"Max latency: {stats['max_latency']:.4f}s")
        print(f"P50 latency: {stats['p50_latency']:.4f}s")
        print(f"P90 latency: {stats['p90_latency']:.4f}s")
        print(f"P95 latency: {stats['p95_latency']:.4f}s")
        print(f"Throughput (tokens/s): {stats['throughput']:.2f}")
    
    def print_summary(self, all_results: Dict):
        """打印总摘要（按并发数升序排序）"""
        print(f"\n{'='*130}")
        print(f"RERANK BENCHMARK SUMMARY REPORT (Model: {self.model_name})")
        print(f"{'='*130}")
        
        print(f"{'Actual Avg Total':<18} {'Top-k':<6} {'Concurrency':<10} {'Success':<8} {'Avg(s)':<10} {'Min(s)':<10} {'Max(s)':<10} {'P90(s)':<10} {'Total Reqs':<8} {'Throughput':<12}")
        print(f"{'-'*130}")
        
        # 核心排序逻辑：先按并发数升序，再按token数、top-k升序
        sorted_keys = sorted(
            all_results.keys(),
            key=lambda x: (int(x.split('conc')[1]), int(x.split('-')[0]), int(x.split('top')[1].split('-')[0]))
        )
        for key in sorted_keys:
            stats = all_results[key]
            print(f"{stats['actual_avg_total_tokens']:<18} {stats['top_k']:<6} {stats['concurrency']:<10} {stats['success_rate']:>7.1%} {stats['avg_latency']:>10.3f} "
                  f"{stats['min_latency']:>10.3f} {stats['max_latency']:>10.3f} {stats['p90_latency']:>10.3f} {stats['total_requests']:<8} {stats['throughput']:>12.2f}")
    
    def save_results(self, all_results: Dict, filename: str = None):
        """保存结果到文件（自动生成带模型名的文件名）"""
        # 自动生成带模型名的文件名
        if filename is None:
            filename = f"rerank_benchmark_{self.model_name}_results.json"
        
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
        """保存关键指标到 CSV（按并发数升序，替换success_rate为success_requests）"""
        # 自动生成带模型名的文件名
        if filename is None:
            filename = f"rerank_benchmark_{self.model_name}_results.csv"
        
        # 调整字段：移除success_rate，新增success_requests
        fieldnames = [
            'actual_tokens',
            'top_k',
            'concurrency',
            'success_requests',  # 替换原success_rate
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
            
            # 核心排序逻辑：先按并发数升序，再按token数、top-k升序
            sorted_keys = sorted(
                all_results.keys(),
                key=lambda x: (
                    int(x.split('conc')[1]),  # 第一优先级：并发数升序（1→2→4）
                    int(x.split('-')[0]),     # 第二优先级：token数升序
                    int(x.split('top')[1].split('-')[0])  # 第三优先级：top-k升序
                )
            )
            
            for key in sorted_keys:
                stats = all_results[key]
                row = {
                    'actual_tokens': stats['actual_avg_total_tokens'],
                    'top_k': stats['top_k'],
                    'concurrency': stats['concurrency'],
                    'success_requests': stats['success_requests'],  # 写入成功请求数
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
    # 核心配置：修复语法错误（原代码少逗号）
    concurrency_config_mapping = {
        1: [
            {"target_total_tokens": 300, "top_k": 5},  # 并发1：测试300token+top5
            {"target_total_tokens": 600, "top_k": 5},  # 并发1：测试600token+top5
            {"target_total_tokens": 1000, "top_k": 5}  # 并发1：测试1000token+top5
        ],
        2: [
            {"target_total_tokens": 600, "top_k": 5},  # 并发2：测试600token+top5
            {"target_total_tokens": 1000, "top_k": 5}  # 并发2：测试1000token+top5
        ],
        4: [
            {"target_total_tokens": 300, "top_k": 5},  # 并发4：测试300token+top5
            {"target_total_tokens": 600, "top_k": 5},  # 并发4：测试600token+top5
            {"target_total_tokens": 1000, "top_k": 5}  # 并发4：测试1000token+top5
        ]
    }
    iterations_per_config = 1  # 每个配置测试1轮
    model_name = "bge-reranker-v2-m3"  # 可切换为 "Qwen3-Reranker-0.6B"
    tei_url = "http://localhost:9997/v1/rerank"
    
    # 创建测试实例（支持指定模型名和URL）
    benchmark = RerankBenchmark(tei_url=tei_url, model_name=model_name)
    
    # 运行测试
    print("Starting rerank benchmark test (Strictly Aligned with Your Calculation Logic)...")
    print(f"Model name: {model_name}")
    print(f"Concurrency-Config mapping: {concurrency_config_mapping}")
    print(f"Iterations per config: {iterations_per_config}")
    
    results = benchmark.run_concurrent_test(
        concurrency_config_mapping=concurrency_config_mapping,
        iterations_per_config=iterations_per_config
    )
    
    # 打印每个组合的详细统计（按并发数排序）
    sorted_keys = sorted(
        results.keys(),
        key=lambda x: (int(x.split('conc')[1]), int(x.split('-')[0]), int(x.split('top')[1].split('-')[0]))
    )
    for key in sorted_keys:
        benchmark.print_stats(results[key])
    
    # 打印总摘要
    benchmark.print_summary(results)
    benchmark.save_results(results)
    benchmark.save_csv(results)

if __name__ == "__main__":
    main()