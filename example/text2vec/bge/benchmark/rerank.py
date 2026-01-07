import time
import requests
import concurrent.futures
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
import json
import csv
import os

class RerankBenchmark:
    def __init__(self, tei_url: str = "http://localhost:9998/v1/rerank", model_name: str = "bge-reranker-v2-m3"):
        self.tei_url = tei_url
        self.model_name = model_name
        self.headers = {"Content-Type": "application/json"}
        self.tokenizer = AutoTokenizer.from_pretrained(f"/FS03/weights/{self.model_name}")
        
        # 固定的基础query和documents模板（按你的要求）
        self.base_query = "A man is eating pasta."
        self.base_documents = [
            "A man is eating food.",
            "A man is eating a piece of bread.",
            "The girl is carrying a baby.",
            "A man is riding a horse.",
            "A woman is playing violin."
        ]
    
    def extend_text_to_token_length(self, text: str, target_tokens: int) -> str:
        """
        将文本重复拓展到目标token长度（精准匹配）
        :param text: 基础文本
        :param target_tokens: 目标token长度（不含特殊token）
        :return: 拓展后的文本
        """
        # 获取基础文本的普通token长度（不含特殊token）
        base_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
        if base_tokens == 0:
            base_tokens = 1
        
        # 计算重复次数
        repeat_times = target_tokens // base_tokens
        remainder = target_tokens % base_tokens
        
        # 基础拓展
        extended_text = text * repeat_times
        
        # 补充剩余token
        if remainder > 0:
            # 从基础文本截取前remainder个token
            base_token_ids = self.tokenizer.encode(text, add_special_tokens=False)[:remainder]
            extended_text += self.tokenizer.decode(base_token_ids, skip_special_tokens=True)
        
        # 最终校准：确保精准匹配目标长度
        actual_tokens = len(self.tokenizer.encode(extended_text, add_special_tokens=False))
        if actual_tokens != target_tokens:
            if actual_tokens > target_tokens:
                # 截取前target_tokens个token
                token_ids = self.tokenizer.encode(extended_text, add_special_tokens=False)[:target_tokens]
                extended_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            else:
                # 补充文本直到长度匹配
                while actual_tokens < target_tokens:
                    extended_text += text[:1]
                    actual_tokens = len(self.tokenizer.encode(extended_text, add_special_tokens=False))
                    if actual_tokens > target_tokens:
                        token_ids = self.tokenizer.encode(extended_text, add_special_tokens=False)[:target_tokens]
                        extended_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                        actual_tokens = target_tokens
        
        return extended_text
    
    def generate_fixed_rerank_data(
        self, 
        target_total_tokens: int,  # 目标总token数（所有pair之和）
        top_k: int = 5  # pair数量
    ) -> Tuple[str, List[str], int]:
        """
        基于固定模板生成rerank测试数据（重复拓展，精准匹配目标总token数）
        """
        # 1. 计算每个pair的目标token数（平均分配）
        target_pair_tokens = max(50, target_total_tokens // top_k)
        # pair的token数 = query_token + doc_token + 2（[CLS]和[SEP]）
        target_pair_normal_tokens = target_pair_tokens - 2
        
        # 2. 拓展query（固定比例：query占20%，doc占80%）
        query_target_tokens = int(target_pair_normal_tokens * 0.2)
        extended_query = self.extend_text_to_token_length(self.base_query, query_target_tokens)
        
        # 3. 拓展documents（每个doc占80%）
        doc_target_tokens = target_pair_normal_tokens - query_target_tokens
        extended_documents = []
        for i, base_doc in enumerate(self.base_documents[:top_k]):
            extended_doc = self.extend_text_to_token_length(base_doc, doc_target_tokens)
            extended_documents.append(extended_doc)
        
        # 4. 计算实际总token数（严格对齐原计算逻辑）
        total_actual_tokens = self.calculate_rerank_input_length(extended_query, extended_documents)
        
        # 5. 最终校准：如果偏差超过5%，微调最后一个doc
        if abs(total_actual_tokens - target_total_tokens) / target_total_tokens > 0.05:
            needed_compensation = target_total_tokens - total_actual_tokens
            last_doc_target = doc_target_tokens + needed_compensation
            if last_doc_target >= 10:  # 避免过短
                extended_documents[-1] = self.extend_text_to_token_length(
                    self.base_documents[min(top_k-1, len(self.base_documents)-1)],
                    last_doc_target
                )
                total_actual_tokens = self.calculate_rerank_input_length(extended_query, extended_documents)
        
        return extended_query, extended_documents, total_actual_tokens
    
    def calculate_rerank_input_length(self, query: str, documents: List[str]) -> int:
        """保持原计算逻辑不变（确保对齐）"""
        total_tokens = 0
        for doc in documents:
            combined_text = f"{query} {doc}"
            tokens = self.tokenizer.encode(combined_text, add_special_tokens=True)
            total_tokens += len(tokens)
        return total_tokens
    
    def warm_up(self, warm_up_config: Dict = None, warm_up_iterations: int = 3):
        """预热Rerank服务（使用固定拓展文本）"""
        print(f"Running warm-up for model: {self.model_name}...")
        if warm_up_config is None:
            warm_up_config = {"target_total_tokens": 500, "top_k": 5}
        
        query, documents, _ = self.generate_fixed_rerank_data(
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
                        "model": self.model_name,
                        "return_documents": False
                    },
                    timeout=300
                )
            except Exception as e:
                print(f"Warm-up error: {e}")
        print("Warm-up completed")
    
    def single_rerank_request(self, query: str, documents: List[str], request_id: int) -> Tuple[int, float, bool]:
        """单个rerank请求函数（逻辑不变）"""
        try:
            total_actual_tokens = self.calculate_rerank_input_length(query, documents)
            start_time = time.perf_counter()
            response = requests.post(
                self.tei_url,
                headers=self.headers,
                json={
                    "query": query,
                    "documents": documents,
                    "model": self.model_name,
                    "return_documents": False
                },
                timeout=600
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
        concurrency_config_mapping: Dict[int, List[Dict]],
        iterations_per_config: int = 5
    ):
        """运行并发rerank测试（逻辑不变，仅替换文本生成函数）"""
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
        
        all_results: Dict[Tuple[int, int, int], List[float]] = {}
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
                
                # 生成固定拓展的测试数据
                query, documents, actual_total = self.generate_fixed_rerank_data(
                    target_total_tokens=target_total_tokens,
                    top_k=top_k
                )
                
                # 打印详细信息
                pair_lengths = []
                for doc in documents:
                    combined_text = f"{query} {doc}"
                    pair_len = len(self.tokenizer.encode(combined_text, add_special_tokens=True))
                    pair_lengths.append(pair_len)
                
                print(f"Target Total Tokens: {target_total_tokens}, Actual Total: {actual_total} (偏差: {abs(actual_total-target_total_tokens)/target_total_tokens:.1%})")
                print(f"Pair Count (Top-k): {top_k}")
                print(f"Each Pair Lengths: {[int(x) for x in pair_lengths]}")
                print(f"Avg Pair Length: {np.mean(pair_lengths):.0f} tokens")
                print(f"Query Length: {len(self.tokenizer.encode(query, add_special_tokens=False))} tokens (普通token)")
                print(f"Concurrency Level: {concurrency}")
                
                # 并发请求
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
        """汇总结果（逻辑不变）"""
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
                    'success_requests': success,
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
        """打印单组统计结果（逻辑不变）"""
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
        """打印总摘要（逻辑不变）"""
        print(f"\n{'='*130}")
        print(f"RERANK BENCHMARK SUMMARY REPORT (Model: {self.model_name}, Fixed Template Text)")
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
    
    def save_results(self, all_results: Dict, filename: str = None, filepath: str = None):
        """保存结果到文件（逻辑不变，增加目录检查）"""
        # 确保保存目录存在
        if filepath and not os.path.exists(filepath):
            os.makedirs(filepath)
        
        # 自动生成带模型名的文件名
        if filename is None:
            filename = f"rerank_benchmark_{self.model_name}_fixed_template_results.json"
        
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
        
        save_path = f"{filepath}/{filename}" if filepath else filename
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {save_path}")

    def save_csv(self, all_results: Dict, filename: str = None, filepath: str = None):
        """保存关键指标到 CSV（逻辑不变，增加目录检查）"""
        # 确保保存目录存在
        if filepath and not os.path.exists(filepath):
            os.makedirs(filepath)
        
        # 自动生成带模型名的文件名
        if filename is None:
            filename = f"rerank_benchmark_{self.model_name}_fixed_template_results.csv"
        
        # 调整字段：移除success_rate，新增success_requests
        fieldnames = [
            'actual_tokens',
            'top_k',
            'concurrency',
            'success_requests',
            'avg_latency',
            'p50_latency',
            'p90_latency',
            'p95_latency',
            'min_latency',
            'max_latency',
            'throughput'
        ]
        
        save_path = f"{filepath}/{filename}" if filepath else filename
        with open(save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # 核心排序逻辑
            sorted_keys = sorted(
                all_results.keys(),
                key=lambda x: (
                    int(x.split('conc')[1]),
                    int(x.split('-')[0]),
                    int(x.split('top')[1].split('-')[0])
                )
            )
            
            for key in sorted_keys:
                stats = all_results[key]
                row = {
                    'actual_tokens': stats['actual_avg_total_tokens'],
                    'top_k': stats['top_k'],
                    'concurrency': stats['concurrency'],
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
        print(f"✅ CSV results saved to {save_path}")

def main():
    # 核心配置：可自定义测试的并发数、目标token数、top-k
    concurrency_config_mapping = {
        2: [
            {"target_total_tokens": 600, "top_k": 5},
            {"target_total_tokens": 1000, "top_k": 5}
        ],
        4: [
            {"target_total_tokens": 300, "top_k": 5},
            {"target_total_tokens": 600, "top_k": 5},
            {"target_total_tokens": 1000, "top_k": 5}
        ]
    }
    iterations_per_config = 1  # 每个配置测试轮数
    model_name = "bge-reranker-v2-m3"
    tei_url = "http://localhost:9999/v1/rerank"
    save_filepath = "./rerank_result"  # 结果保存目录
    
    benchmark = RerankBenchmark(tei_url=tei_url, model_name=model_name)
    
    # 运行测试
    print(f"Model name: {model_name}")
    print(f"Concurrency-Config mapping: {concurrency_config_mapping}")
    print(f"Iterations per config: {iterations_per_config}")
    
    results = benchmark.run_concurrent_test(
        concurrency_config_mapping=concurrency_config_mapping,
        iterations_per_config=iterations_per_config
    )
    
    # 打印详细统计和总摘要
    sorted_keys = sorted(
        results.keys(),
        key=lambda x: (int(x.split('conc')[1]), int(x.split('-')[0]), int(x.split('top')[1].split('-')[0]))
    )
    for key in sorted_keys:
        benchmark.print_stats(results[key])
    
    benchmark.print_summary(results)

    # 保存结果
    benchmark.save_results(results, filepath=save_filepath)
    benchmark.save_csv(results, filepath=save_filepath)
    

if __name__ == "__main__":
    main()
