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

def launch_ds3_model(args: argparse):
    base_url = args.url

    client = RESTfulClient(base_url)
    with open(args.model_config, 'r', encoding='utf-8') as f:
        model = f.read()
    model_dict = json.loads(model)
    model_path = model_dict.get("model_path", "/weights/Deepseek-V3-0324")
    model_uid = client.launch_model(
        model_name="deepseek-r1",
        model_uid="deepseek-r1",
        model_engine="vllm",
        model_size_in_billions=671,
        model_path=model_path,
        n_gpu=args.n_gpu,
        replica=args.instance_nums,
        reasoning_content=True,
        **model_dict.get("additional_params", {})
    )
    print(model_uid)
    #llm_model = client.get_model(model_uid)
    #response = llm_model.generate("What is the largest animal in the world?")
    #print(response)
    return client, model_uid

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
                        default=32,
                        help="gpu nums")
    args = parser.parse_args()
    
    launch_ds3_model(args)

