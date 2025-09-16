import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import psutil
import requests
import numpy as np
import aiohttp
import asyncio
from typing import *
from loguru import logger
from transformers import AutoTokenizer
import random
import argparse

from xinference.client.restful.restful_client import Client as RESTfulClient
from xinference.client.restful.restful_client import (
    RESTfulChatModelHandle,
    RESTfulEmbeddingModelHandle,
    _get_error_string,
)

def launch_custom_embedding_model(args: argparse):
    base_url = args.url

    client = RESTfulClient(base_url)

    model_regs = client.list_model_registrations(model_type="embedding")
    assert len(model_regs) > 0

    with open(args.model_config, 'r', encoding='utf-8') as f:
        model = f.read()
    client.register_model(model_type="embedding", model=model, persist=True)
    
    model_dict = json.loads(model)
    model_name = model_dict.get('model_name', 'custom-embedding')
    data = client.get_model_registration(model_type="embedding", model_name=model_name)
    print(f"data:{data}")
    assert model_name in data["model_name"]
    
    
    new_model_regs = client.list_model_registrations(model_type="embedding")
    assert len(new_model_regs) == len(model_regs) + 1
    custom_model_reg = None
    for model_reg in new_model_regs:
        if model_reg["model_name"] == model_name:
            custom_model_reg = model_reg
    assert custom_model_reg is not None
    model_uid = client.launch_model(
        model_name=model_name,
        model_type="embedding",
        model_uid=model_name,
        replica=args.instance_nums,
        n_gpu=args.n_gpu
    )
    print(model_uid)
    ##发请求验证
    embedding_model = client.get_model(model_uid)
    sentence = ["What is the capital of China?"]
    print(embedding_model.create_embedding(sentence))
    return client, model_uid


def launch_custom_rerank_model(args: argparse):
    base_url = args.url

    client = RESTfulClient(base_url)

    model_regs = client.list_model_registrations(model_type="rerank")
    assert len(model_regs) > 0

    with open(args.model_config, 'r', encoding='utf-8') as f:
        model = f.read()
    client.register_model(model_type="rerank", model=model, persist=True)
    
    model_dict = json.loads(model)
    model_name = model_dict.get('model_name', 'custom-rerank')
    data = client.get_model_registration(model_type="rerank", model_name=model_name)
    print(f"data:{data}")
    assert model_name in data["model_name"]
    
    
    new_model_regs = client.list_model_registrations(model_type="rerank")
    assert len(new_model_regs) == len(model_regs) + 1
    custom_model_reg = None
    for model_reg in new_model_regs:
        if model_reg["model_name"] == model_name:
            custom_model_reg = model_reg
    assert custom_model_reg is not None
    model_uid = client.launch_model(
        model_name=model_name,
        model_type="rerank",
        model_uid=model_name,
        replica=args.instance_nums
    )
    print(model_uid)
    rerank_model = client.get_model(model_uid)
    query = "A man is eating pasta."
    corpus = [
    "A man is eating food.",
    "A man is eating a piece of bread.",
    "The girl is carrying a baby.",
    "A man is riding a horse.",
    "A woman is playing violin."
    ]
    print(rerank_model.rerank(corpus, query))
    return client, model_uid

# (prompt len, output len, latency)
REQUEST_LATENCY: List[Tuple[int, int, float]] = []

def launch_custom_model(args: argparse):
    base_url = args.url

    client = RESTfulClient(base_url)

    model_regs = client.list_model_registrations(model_type="LLM")
    assert len(model_regs) > 0

    with open(args.model_config, 'r', encoding='utf-8') as f:
        model = f.read()
    client.register_model(model_type="LLM", model=model, persist=False)
    
    model_dict = json.loads(model)
    model_name = model_dict.get('model_name', 'custom-llm')
    model_specs = model_dict.get('model_specs', {})

    model_size_in_billions = int(model_specs[0].get('model_size_in_billions', 7)) 
    data = client.get_model_registration(model_type="LLM", model_name=model_name)
    print(f"data:{data}")
    assert model_name in data["model_name"]
    
    
    new_model_regs = client.list_model_registrations(model_type="LLM")
    assert len(new_model_regs) == len(model_regs) + 1
    custom_model_reg = None
    for model_reg in new_model_regs:
        if model_reg["model_name"] == model_name:
            custom_model_reg = model_reg
    assert custom_model_reg is not None
    model_uid = client.launch_model(
        model_name=model_name,
        #model_engine="Transformers",
        model_engine="vastai",
        model_uid=model_name,
        model_size_in_billions=model_size_in_billions,
        quantization="none",
        replica=args.instance_nums
    )
    print(model_uid)
    # llm_model = client.get_model(model_uid)
    # response = llm_model.generate("What is the largest animal in the world?")
    # print(response)
    return client, model_uid

def test_custom_model_benchmark(args: argparse):
    base_url = args.url
    tokenizer_path = args.tokenizer
    dataset_path = args.dataset
    
    client, model_uid = launch_custom_model(args)
    
    model = client.get_model(model_uid=model_uid)
    assert isinstance(model, RESTfulChatModelHandle)
    completion = model.generate(
        "Once upon a time, there was a very old computer", {"max_tokens": 64}
    )
    assert "text" in completion["choices"][0]
    

    completion = model.chat("What is the capital of France?")
    assert "content" in completion["choices"][0]["message"]

    def _check_stream():
        streaming_response = model.chat(
            prompt="What is the capital of France?",
            generate_config={"stream": True, "max_tokens": 5},
        )
        for chunk in streaming_response:
            assert ("content" in chunk["choices"][0]["delta"]) or (
                "role" in chunk["choices"][0]["delta"]
            )

    _check_stream()

    results = []
    with ThreadPoolExecutor() as executor:
        for _ in range(2):
            r = executor.submit(_check_stream)
            results.append(r)

    # After iteration finish, we can iterate again.
    _check_stream()

    test_benchmark(base_url, tokenizer_path, dataset_path ,num_prompts=args.num_prompts)

    # client.terminate_model(model_uid=model_uid)

    
    

async def get_request(input_requests: List[Tuple[str, int, int]],
                        request_rate: float,
                    ) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)

async def send_request(
    api_url: str,
    prompt: str,
    prompt_len: int,
    output_len: int
) -> None:
    request_start_time = time.perf_counter()

    headers = {"User-Agent": "Benchmark Client"}
    pload = {
        "model": "custom-llm",
        "prompt": prompt,
        "temperature": 0.0 ,
        "top_p": 1.0,
        "max_tokens": output_len,
        "ignore_eos": True,
        "stream": False,
    }
  

    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(api_url, headers=headers, json=pload) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")
            output = json.loads(output)

            # Re-send the request if it failed.
            if "error" not in output:
                break

    request_end_time = time.perf_counter()
    request_latency = request_end_time - request_start_time
    REQUEST_LATENCY.append((prompt_len, output_len, request_latency))


def sample_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer,
    fixed_output_len: Optional[int],
) -> List[Tuple[str, int, int]]:
    if fixed_output_len is not None and fixed_output_len < 4:
        raise ValueError("output_len too small")

    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2][:2000]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        if fixed_output_len is not None:
            output_len = fixed_output_len
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))
    
    # Sample the requests.
    sampled_requests = filtered_dataset[:num_requests] #random.sample(filtered_dataset, num_requests)
    return sampled_requests


async def benchmark(
    api_url: str,
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> None:
    tasks: List[asyncio.Task] = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        task = asyncio.create_task(send_request(api_url, prompt, prompt_len, output_len))
        tasks.append(task)
    await asyncio.gather(*tasks)


def test_benchmark(base_url, tokenizer_path, dataset_path, input_len=512, output_len=512, num_prompts=200):
    api_url = f"{base_url}/v1/completions"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    input_requests = sample_requests(dataset_path, num_prompts, tokenizer, None)
    # random.shuffle(input_requests)
    
    # input_requests = [("hi "*(input_len-3), input_len, output_len) for _ in range(num_prompts)]
    REQUEST_LATENCY.clear()

    benchmark_start_time = time.perf_counter()
    asyncio.run(benchmark(api_url, input_requests, request_rate = float("inf")))
    benchmark_end_time = time.perf_counter()
    benchmark_time = benchmark_end_time - benchmark_start_time
    print(f"Total time: {benchmark_time:.2f} s")
    print(f"Throughput: {num_prompts / benchmark_time:.2f} requests/s")

    # Compute the latency statistics.
    avg_latency = np.mean([latency for _, _, latency in REQUEST_LATENCY])
    print(f"Average latency: {avg_latency:.2f} s")
    avg_per_token_latency = np.mean([
        latency / (prompt_len + output_len)
        for prompt_len, output_len, latency in REQUEST_LATENCY
    ])
    print(f"Average latency per token: {avg_per_token_latency:.2f} s")
    avg_per_output_token_latency = np.mean([
        latency / output_len
        for _, output_len, latency in REQUEST_LATENCY
    ])
    print("Average latency per output token: "
          f"{avg_per_output_token_latency:.2f} s")
    throughput = (
        sum([output_len for _, output_len, _ in REQUEST_LATENCY]) / benchmark_time
    )
    print(f"Throughput: {throughput} tokens/s, \
      All input tokens: {sum([input_len for input_len, _, _ in REQUEST_LATENCY])}, \
      All generate tokens: {sum([output_len for _, output_len, _ in REQUEST_LATENCY])}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test Client.")
    parser.add_argument("--url", type=str, default='http://192.168.25.141:9997')
    parser.add_argument("--model-config",
                        type=str,
                        default='/home/mqxu/work/project/_tmp/xinference_modify/docs/register_model.json',
                        help="model tokenizer path")
    parser.add_argument("--instance-nums",
                        type=int,
                        default=2,
                        help="model instance nums")                    
    parser.add_argument("--n_gpu",
                        type=int,
                        default=2,
                        help="gpu nums")
    parser.add_argument('--test-benchmark',
                        action='store_true',
                        help='test benchmark')
    parser.add_argument('--embedding',
                        action='store_true',
                        help='test embedding')
    parser.add_argument('--rerank',
                        action='store_true',
                        help='test rerank')
    parser.add_argument("--tokenizer",
                        type=str,
                        default='/home/mqxu/work/project/_tmp/ai21rc1/llama2_7b-fp16-none-dynamic-vacc/tokenizer',
                        help="model tokenizer path")
    parser.add_argument("--dataset",
                        type=str,
                        default='/home/mqxu/work/project/_tmp/release/ShareGPT_V3_unfiltered_cleaned_split.json',
                        help="dataset path")
    parser.add_argument("--num-prompts",
                        type=int,
                        default=100,
                        help="Number of prompts to process.")
    args = parser.parse_args()
    
    if args.test_benchmark:
        test_custom_model_benchmark(args)
    elif args.embedding:
        launch_custom_embedding_model(args)
    elif args.rerank:
        launch_custom_rerank_model(args)
    else:
        launch_custom_model(args)

