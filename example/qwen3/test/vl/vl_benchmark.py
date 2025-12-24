import requests
import json
import base64
import argparse
import time
import concurrent.futures
import numpy as np
from typing import List, Dict, Tuple
import csv
import re
import random
import string
from PIL import Image
import io

def generate_fixed_length_prompt(token_length: int) -> str:
    """
    生成指定Token长度的测试Prompt（近似Token数）
    :param token_length: 目标Token长度
    :return: 固定Token长度的文本Prompt
    """
    base_chars = string.ascii_letters + string.digits + "，。！？；："
    prompt = ""
    current_tokens = 0
    
    while current_tokens < token_length:
        if random.random() > 0.5:
            char = random.choice(["的", "了", "是", "在", "有", "我", "他", "你", "这", "那"])
            prompt += char
            current_tokens += 1.5
        else:
            char = random.choice(base_chars)
            prompt += char
            current_tokens += 0.3
    
    prompt = f"详细描述这张图片的内容，要求回答长度不少于{token_length}个Token：{prompt[:int(token_length*0.8)]}"
    return prompt.strip()

def generate_fixed_size_image(width: int, height: int, image_format: str = "jpg") -> str:
    """生成指定尺寸的空白测试图片（Base64格式）"""
    pil_format = "JPEG" if image_format.lower() == "jpg" else image_format.upper()
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    
    # 添加随机噪点模拟真实图片
    pixels = img.load()
    for i in range(width):
        for j in range(height):
            if random.random() < 0.05:
                pixels[i, j] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    img_buffer = io.BytesIO()
    img.save(img_buffer, format=pil_format)
    img_buffer.seek(0)
    
    base64_str = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    return f"data:image/{image_format};base64,{base64_str}"

def image_file_to_fixed_size_base64(image_path: str, target_width: int, target_height: int) -> str:
    """将本地图片缩放为指定尺寸并转为Base64"""
    try:
        with Image.open(image_path) as img:
            img.thumbnail((target_width, target_height))
            new_img = Image.new('RGB', (target_width, target_height), (255, 255, 255))
            paste_x = (target_width - img.width) // 2
            paste_y = (target_height - img.height) // 2
            new_img.paste(img, (paste_x, paste_y))
            
            img_buffer = io.BytesIO()
            img_format = image_path.split(".")[-1].lower()
            pil_format = "JPEG" if img_format == "jpg" else img_format.upper()
            new_img.save(img_buffer, format=pil_format)
            img_buffer.seek(0)
            
            base64_str = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
            return f"data:image/{img_format};base64,{base64_str}"
    except Exception as e:
        raise Exception(f"处理图片失败: {str(e)}")

class Qwen3VLBenchmark:
    def __init__(self, model_name: str, port: int, resolution: str, 
                 random_input_len: int, random_output_len: int,
                 use_local_image: bool = False, local_image_path: str = None):
        # 基础配置
        self.model_name = model_name
        self.port = port
        self.resolution = resolution  # 如 "1280x720"
        self.random_input_len = random_input_len  # Prompt Token长度
        self.random_output_len = random_output_len  # 期望输出Token长度
        self.use_local_image = use_local_image
        self.local_image_path = local_image_path
        
        # 解析分辨率
        self.img_width, self.img_height = map(int, resolution.split("x"))
        
        # 请求配置
        self.url = f"http://localhost:{port}/v1/chat/completions"
        self.headers = {"Content-Type": "application/json"}
        
        # 生成Prompt和图片
        self.prompt = generate_fixed_length_prompt(random_input_len)
        if use_local_image and local_image_path:
            self.base64_image = image_file_to_fixed_size_base64(
                local_image_path, self.img_width, self.img_height
            )
        else:
            self.base64_image = generate_fixed_size_image(self.img_width, self.img_height)
        
        # 构建请求体
        self.base_content = [{"type": "text", "text": self.prompt}]
        if self.base64_image:
            self.base_content.append({
                "type": "image_url",
                "image_url": {"url": self.base64_image}
            })
        
        # 测试结果存储
        self.test_results: List[Dict] = []

    def parse_stream_response(self, response) -> Tuple[float, float, str, int]:
        """解析流式响应，精准计算TTFT和各指标"""
        ttft = -1.0  # 首Token时间（秒）
        decode_latency = -1.0  # 解码总耗时（首Token到最后Token）
        full_response = ""
        token_count = 0
        first_token_received = False
        stream_start_time = time.perf_counter()
        last_token_time = stream_start_time

        try:
            for chunk in response.iter_lines():
                if chunk:
                    chunk = chunk.decode('utf-8').strip()
                    if chunk == "data: [DONE]" or not chunk.startswith("data: "):
                        continue
                    
                    try:
                        data = json.loads(chunk[6:])
                        if "choices" in data and len(data["choices"]) > 0:
                            choice = data["choices"][0]
                            delta = choice.get("delta", {})
                            content = delta.get("content", "")
                            
                            if content and not first_token_received:
                                ttft = time.perf_counter() - stream_start_time
                                first_token_received = True
                                token_count += 1
                                full_response += content
                            elif content:
                                token_count += 1
                                full_response += content
                            
                            if first_token_received:
                                last_token_time = time.perf_counter()
                    except json.JSONDecodeError:
                        continue
            
            if first_token_received:
                decode_latency = last_token_time - (stream_start_time + ttft)
            
            return ttft, decode_latency, full_response.strip(), token_count
        except Exception as e:
            return -1.0, -1.0, f"解析失败: {str(e)}", 0

    def single_request(self, request_id: int) -> Dict:
        """单个VL流式请求，返回标准化指标"""
        result = {
            "request_id": request_id,
            "success": False,
            "ttft_ms": -1.0,          # Mean TTFT (ms)
            "tpot_ms": -1.0,          # Mean TPOT (ms) - Token Per Output Token
            "decoding_token_throughput": -1.0,  # 解码Token吞吐量
            "per_req_decoding_throughput": -1.0, # 单请求解码吞吐量
            "total_latency_ms": -1.0,
            "decode_latency_ms": -1.0,
            "token_count": 0,
            "error_msg": ""
        }

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": self.base_content}],
            "temperature": 0.7,
            "max_tokens": self.random_output_len,
            "stream": True
        }

        try:
            total_start = time.perf_counter()
            response = requests.post(
                self.url,
                headers=self.headers,
                data=json.dumps(payload),
                timeout=1200,
                stream=True
            )
            response.raise_for_status()
            
            # 解析流式响应
            ttft, decode_latency, _, token_count = self.parse_stream_response(response)
            total_end = time.perf_counter()
            
            # 转换为毫秒级指标
            if ttft > 0 and decode_latency > 0 and token_count > 0:
                result["success"] = True
                result["ttft_ms"] = round(ttft * 1000, 2)  # 转换为毫秒
                result["total_latency_ms"] = round((total_end - total_start) * 1000, 2)
                result["decode_latency_ms"] = round(decode_latency * 1000, 2)
                result["token_count"] = token_count
                
                # 计算核心指标
                result["tpot_ms"] = round(result["decode_latency_ms"] / token_count, 2)  # TPOT = 解码总耗时 / Token数
                result["decoding_token_throughput"] = round(token_count / (decode_latency), 2)  # Token/秒
                result["per_req_decoding_throughput"] = result["decoding_token_throughput"]  # 单请求吞吐量

        except requests.exceptions.ConnectionError:
            result["error_msg"] = "服务连接失败"
        except requests.exceptions.Timeout:
            result["error_msg"] = "请求超时"
        except Exception as e:
            result["error_msg"] = f"请求失败: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                result["error_msg"] += f" | {e.response.text}"
        
        return result

    def warm_up(self, warm_up_times: int = 3):
        """预热模型"""
        print(f"\n=== 预热 {self.model_name} 模型 ===")
        for i in range(warm_up_times):
            try:
                self.single_request(f"warmup_{i}")
                print(f"预热请求 {i+1}/{warm_up_times} 完成")
            except Exception as e:
                print(f"预热请求 {i+1}/{warm_up_times} 失败: {e}")
        print("=== 预热完成 ===\n")

    def run_test(self, requests_count: int, concurrency: int) -> Dict:
        """
        运行指定并发和请求数的测试
        :param requests_count: 总请求数（对应表格的requests列）
        :param concurrency: 并发数
        :return: 汇总统计结果
        """
        # 预热
        self.warm_up()
        
        print(f"\n=== 开始测试 ===")
        print(f"模型: {self.model_name} | 分辨率: {self.resolution}")
        print(f"请求数: {requests_count} | 并发数: {concurrency}")
        print(f"输入Token长度: {self.random_input_len} | 输出Token长度: {self.random_output_len}")
        
        # 并发执行请求
        self.test_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for req_id in range(requests_count):
                future = executor.submit(self.single_request, req_id)
                futures.append(future)
            
            # 收集结果
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                req_result = future.result()
                self.test_results.append(req_result)
                completed += 1
                status = "成功" if req_result['success'] else "失败"
                print(f"完成 {completed}/{requests_count} 请求 | ID: {req_result['request_id']} | 状态: {status}")
        
        # 统计有效结果
        valid_results = [r for r in self.test_results if r["success"]]
        if not valid_results:
            print("⚠️  无有效测试结果")
            return {}
        
        # 计算均值指标
        avg_ttft_ms = round(np.mean([r["ttft_ms"] for r in valid_results]), 2)
        avg_tpot_ms = round(np.mean([r["tpot_ms"] for r in valid_results]), 2)
        avg_decoding_throughput = round(np.mean([r["decoding_token_throughput"] for r in valid_results]), 2)
        avg_per_req_throughput = round(avg_decoding_throughput / concurrency, 2)  # 按并发数均分
        
        # 构建标准化输出结果
        summary = {
            "resolution": self.resolution,
            "requests": requests_count,
            "concurrency": concurrency,
            "random-input-len": self.random_input_len,
            "random-output-len": self.random_output_len,
            "Mean TTFT (ms)": avg_ttft_ms,
            "Mean TPOT (ms)": avg_tpot_ms,
            "decoding_token_throughput": avg_decoding_throughput,
            "per_req_decoding_throughput": avg_per_req_throughput
        }
        
        print(f"\n=== 测试完成 - 统计结果 ===")
        for k, v in summary.items():
            print(f"{k}: {v}")
        
        return summary

def main():
    parser = argparse.ArgumentParser(description="Qwen3-VL 标准化性能测试工具")
    # 模型基础配置
    parser.add_argument("--model-name", required=True, help="模型名称（如 Qwen3-VL-30B-A3B-Instruct-FP8）")
    parser.add_argument("--port", default=9992, type=int, help="服务端口")
    # 测试维度配置（与输出表格一一对应）
    parser.add_argument("--resolution", required=True, help="图片分辨率（如 1280x720）")
    parser.add_argument("--requests", required=True, type=int, help="总请求数（如 20）")
    parser.add_argument("--concurrency", required=True, type=int, help="并发数（如 1/2/4）")
    parser.add_argument("--random-input-len", required=True, type=int, help="输入Prompt Token长度（如 128）")
    parser.add_argument("--random-output-len", required=True, type=int, help="期望输出Token长度（如 1024）")
    # 图片配置
    parser.add_argument("--use-local-image", action="store_true", help="是否使用本地图片")
    parser.add_argument("--local-image-path", default=None, help="本地图片路径")
    # 输出配置
    parser.add_argument("--output", default="qwen3_vl_perf_results.csv", help="CSV输出文件名")
    
    args = parser.parse_args()

    # 校验参数
    if args.use_local_image and not args.local_image_path:
        parser.error("--use-local-image 需要配合 --local-image-path 指定本地图片路径")

    # 创建测试实例
    benchmark = Qwen3VLBenchmark(
        model_name=args.model_name,
        port=args.port,
        resolution=args.resolution,
        random_input_len=args.random_input_len,
        random_output_len=args.random_output_len,
        use_local_image=args.use_local_image,
        local_image_path=args.local_image_path
    )

    # 运行测试
    test_summary = benchmark.run_test(
        requests_count=args.requests,
        concurrency=args.concurrency
    )

    # 写入CSV文件（追加模式，支持多次测试）
    if test_summary:
        # 定义CSV表头（与你提供的格式完全一致）
        fieldnames = [
            "resolution", "requests", "concurrency", "random-input-len", 
            "random-output-len", "Mean TTFT (ms)", "Mean TPOT (ms)", 
            "decoding_token_throughput", "per_req_decoding_throughput"
        ]
        
        # 检查文件是否存在，不存在则写入表头
        import os
        file_exists = os.path.exists(args.output)
        
        with open(args.output, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            # 写入当前测试结果
            writer.writerow(test_summary)
        
        print(f"\n✅ 测试结果已写入: {args.output}")

if __name__ == "__main__":
    main()