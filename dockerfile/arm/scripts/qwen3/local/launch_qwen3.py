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

def launch_qwen3_model(args: argparse, gpu_list: list[int]):
    base_url = args.url

    client = RESTfulClient(base_url)
    with open(args.model_config, 'r', encoding='utf-8') as f:
        model = f.read()
    model_dict = json.loads(model)
    model_path = model_dict.get("model_path", "/logs/weights/Qwen3-30B-A3B-FP8")
    if args.think:
        model_uid = client.launch_model(
            model_name="qwen3",
            model_uid="qwen3",
            model_engine="vllm",
            model_size_in_billions=args.billion,
            model_path=model_path,
            n_gpu=args.n_gpu,
            replica=args.instance_nums,
            gpu_idx=gpu_list,
            enable_thinking=True,
            reasoning_content=True,
            **model_dict.get("additional_params", {})
        )
        print("launch with thinking mode")
    else:
        model_uid = client.launch_model(
            model_name="qwen3",
            model_uid="qwen3",
            model_engine="vllm",
            model_size_in_billions=args.billion,
            model_path=model_path,
            n_gpu=args.n_gpu,
            replica=args.instance_nums,
            gpu_idx=gpu_list,
            **model_dict.get("additional_params", {})
        )
        print("launchout with thinking mode")
    print(model_uid)
    #llm_model = client.get_model(model_uid)
    #response = llm_model.generate("What is the largest animal in the world?")
    #print(response)
    return client, model_uid

def parse_GPU_LIST(gpu_str: str) -> list[int]:
    """将逗号分隔的字符串转为整数列表"""
    try:
        # 分割后转整数，过滤空值（防止末尾有逗号）
        return [int(gpu_id.strip()) for gpu_id in gpu_str.split(',') if gpu_id.strip()]
    except ValueError as e:
        raise ValueError(f"GPU_LIST 格式错误，需为逗号分隔的整数：{e}")

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
    parser.add_argument("--billion",
                    type=int,
                    default=30,
                    help="size")
    parser.add_argument("--think",
                    type=bool,
                    default=True,
                    help="enable thinking")
    parser.add_argument("--gpu_idx",
                    type=str,
                    default="0,1,2,3",
                    help="gpu idx")
    args = parser.parse_args()

    # 转为整数列表
    gpu_list = parse_GPU_LIST(args.gpu_idx)

    launch_qwen3_model(args,gpu_list)
