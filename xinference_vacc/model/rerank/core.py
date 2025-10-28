# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import importlib
import logging
import os
import threading
import uuid
from collections import defaultdict
from collections.abc import Sequence
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from xinference.model.rerank.core import RerankModelSpec, RerankModelDescription
from xinference.types import Document, DocumentObj, Rerank, RerankTokens
from xinference.device_utils import empty_cache
import json
from transformers import AutoTokenizer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)
# Used for check whether the model is cached.
# Init when registering all the builtin models.
MODEL_NAME_TO_REVISION: Dict[str, List[str]] = defaultdict(list)
RERANK_MODEL_DESCRIPTIONS: Dict[str, List[Dict]] = defaultdict(list)
RERANK_EMPTY_CACHE_COUNT = int(os.getenv("XINFERENCE_RERANK_EMPTY_CACHE_COUNT", "10"))
assert RERANK_EMPTY_CACHE_COUNT > 0

class RerankModel:
    class ModelInfo:
        def __init__(self, model_path: str, batch_size: int, max_seqlen: int):
            self.model_path = model_path
            self.batch_size = batch_size
            self.max_seqlen = max_seqlen
            self._model = None  # 初始化为 None
            self.vsx_engine = None 
    def __init__(
        self,
        model_spec: RerankModelSpec,
        model_uid: str,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        use_fp16: bool = False,
        model_config: Optional[Dict] = None,
    ):
        self._model_spec = model_spec
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        self._model_config = model_config or dict()
        self._use_fp16 = use_fp16
        self._model = None
        self._counter = 0
        logger.info(f"RerankModel model_spec:{model_spec}")
        self.vastai_rerank = False
        logger.info(f"self._model_path:{self._model_path}")
        self.modelInfo_list = []
        self.model_info_dict = {}
        ##加载三件套
        if self._model_path.endswith("mod"):
            self.vastai_rerank = True
            self._model_spec.type = "vastai"
            self.root_dir = self._model_path[:-3]
            subdirs_list = self._find_model_subdirs()
            self._model_tokenizer = self.root_dir + subdirs_list[0] + "/tokenizer"
            logger.info(f"Tokenizer path: {self._model_tokenizer}")
            # 加载一次 tokenizer
            try:
                self.vastai_tokenizer = AutoTokenizer.from_pretrained(self._model_tokenizer)
            except Exception as e:
                logger.error(f"Failed to load tokenizer: {e}")
                self.vastai_tokenizer = None
        ##如果还有子目录 512,1024,2048,4096,8192

            for i in subdirs_list:
                # 如果有子目录
                subdir_modpath = os.path.join(self.root_dir, i, "mod")
                logger.info(f"subdirectory mod path: {subdir_modpath}")
                model_info = self._initialize_vastai_model(subdir_modpath)
                self.modelInfo_list.append(model_info)
            logger.info(f"目前modelInfo list 长度: {len(self.modelInfo_list)}")
        ##如果是非vastai rerank 再去检测类型
        if not self.vastai_rerank:
            if model_spec.type == "unknown":
                model_spec.type = self._auto_detect_type(model_path)

    def _initialize_vastai_model(self, model_path: Optional[str] = None) -> ModelInfo:
        """初始化 vastai 格式的模型"""
        if model_path is None:
            model_path = self._model_path
        
        # 默认值
        batch_size = 1
        max_seqlen = 512
        
        # 读取配置文件
        vacc_config_path = os.path.join(model_path[:-3] + 'tokenizer', "vacc_config.json")
        if os.path.exists(vacc_config_path):
            try:
                with open(vacc_config_path, 'r') as f:
                    config = json.load(f)
                batch_size = config.get('batch_size', 1)
                max_seqlen = config.get('max_seqlen', 512)
                logger.info(f"Read rerank config from {vacc_config_path}")
                logger.info(f"Loaded config - batch size: {batch_size}, max_seqlen: {max_seqlen}")
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON format in config file: {e}")
            except Exception as e:
                logger.error(f"Error reading config file: {e}")
        else:
            logger.info(f"Using default config - batch size: {batch_size}, max_seqlen: {max_seqlen}")
        # 创建并返回 ModelInfo
        return self.ModelInfo(model_path, batch_size, max_seqlen)

    def _find_model_subdirs(self) -> list:
        """查找模型目录下的数字子目录（如1024、2048等）"""
        subdirs = []
        if os.path.isdir(self.root_dir):
            for item in os.listdir(self.root_dir):
                item_path = os.path.join(self.root_dir, item)
                if os.path.isdir(item_path) and item.isdigit():
                    subdirs.append(item)
        # 按数字大小排序
        subdirs.sort(key=int)
        return subdirs

    @staticmethod
    def _get_tokenizer(model_path):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return tokenizer

    @staticmethod
    def _auto_detect_type(model_path):
        """This method may not be stable due to the fact that the tokenizer name may be changed.
        Therefore, we only use this method for unknown model types."""

        type_mapper = {
            "LlamaTokenizerFast": "LLM-based layerwise",
            "GemmaTokenizerFast": "LLM-based",
            "XLMRobertaTokenizerFast": "normal",
        }

        tokenizer = RerankModel._get_tokenizer(model_path)
        rerank_type = type_mapper.get(type(tokenizer).__name__)
        if rerank_type is None:
            logger.warning(
                f"Can't determine the rerank type based on the tokenizer {tokenizer}, use normal type by default."
            )
            return "normal"
        return rerank_type

    def load(self):
        if self.vastai_rerank and self._model_spec.type == "vastai":
            logger.info("now loading vastai rerank")
            self.vastai_rerank_id = int(self._device)
            logger.info(f"self.vastai_rerank_id:{self.vastai_rerank_id}")
            from ..embedding.vastai_core import VastaiReranker
            for model_info in self.modelInfo_list:
                model_info._model = VastaiReranker(
                    model_info.model_path,
                    self._model_tokenizer,
                    device_id=self.vastai_rerank_id,
                    batch_size=model_info.batch_size,
                    max_seqlen=model_info.max_seqlen,
                )
                logger.info(f"rerank model info:{model_info.model_path}, {model_info.batch_size}, {model_info.max_seqlen}")
                model_info.vsx_engine = model_info._model.init_vaststreamx()
                #{512：model_info, 1024:model_info}
                self.model_info_dict[model_info.max_seqlen] = model_info
            logger.info(f"finish loading vastai rerank id:{self.vastai_rerank_id}")
            return
        logger.info("Loading rerank model: %s", self._model_path)
        flash_attn_installed = importlib.util.find_spec("flash_attn") is not None
        if (
            self._auto_detect_type(self._model_path) != "normal"
            and flash_attn_installed
        ):
            logger.warning(
                "flash_attn can only support fp16 and bf16, "
                "will force set `use_fp16` to True"
            )
            self._use_fp16 = True

        if self._model_spec.type == "normal":
            try:
                import sentence_transformers
                from sentence_transformers.cross_encoder import CrossEncoder

                if sentence_transformers.__version__ < "3.1.0":
                    raise ValueError(
                        "The sentence_transformers version must be greater than 3.1.0. "
                        "Please upgrade your version via `pip install -U sentence_transformers` or refer to "
                        "https://github.com/UKPLab/sentence-transformers"
                    )
            except ImportError:
                error_message = "Failed to import module 'sentence-transformers'"
                installation_guide = [
                    "Please make sure 'sentence-transformers' is installed. ",
                    "You can install it by `pip install sentence-transformers`\n",
                ]

                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
            self._model = CrossEncoder(
                self._model_path,
                device=self._device,
                trust_remote_code=True,
                max_length=getattr(self._model_spec, "max_tokens"),
                **self._model_config,
            )
            if self._use_fp16:
                self._model.model.half()
        else:
            try:
                if self._model_spec.type == "LLM-based":
                    from FlagEmbedding import FlagLLMReranker as FlagReranker
                elif self._model_spec.type == "LLM-based layerwise":
                    from FlagEmbedding import LayerWiseFlagLLMReranker as FlagReranker
                else:
                    raise RuntimeError(
                        f"Unsupported Rank model type: {self._model_spec.type}"
                    )
            except ImportError:
                error_message = "Failed to import module 'FlagEmbedding'"
                installation_guide = [
                    "Please make sure 'FlagEmbedding' is installed. ",
                    "You can install it by `pip install FlagEmbedding`\n",
                ]

                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
            self._model = FlagReranker(self._model_path, use_fp16=self._use_fp16)
        # Wrap transformers model to record number of tokens
        self._model.model = _ModelWrapper(self._model.model)

    def calculate_rerank_input_length(self, query: str, documents: List[str]):
        input_lengths = []
        for doc in documents:
            # 通常的格式: [CLS] query [SEP] document [SEP]
            combined_text = f"{query} {doc}"
            tokens = self.vastai_tokenizer.encode(combined_text, add_special_tokens=True)
            input_lengths.append(len(tokens))
        return sum(input_lengths)

    def rerank(
        self,
        documents: List[str],
        query: str,
        top_n: Optional[int],
        max_chunks_per_doc: Optional[int],
        return_documents: Optional[bool],
        return_len: Optional[bool],
        **kwargs,
    ) -> Rerank:
        if max_chunks_per_doc is not None:
            raise ValueError("rerank hasn't support `max_chunks_per_doc` parameter.")
        logger.info("Rerank with kwargs: %s, model: %s", kwargs, self._model)
        sentence_combinations = [[query, doc] for doc in documents]
        # reset n tokens
        if self._model_spec.type != "vastai":
            self._model.model.n_tokens = 0
        if self._model_spec.type == "vastai":
            all_token_nums = self.calculate_rerank_input_length(query, documents)
            logger.info(f"rerank input length:{all_token_nums}")
            # 获取所有整数类型的max_seqlen（过滤掉字符串key）
            available_seq_lens = [ max_seqlen for max_seqlen in self.model_info_dict.keys() ]
            # 排序
            available_seq_lens.sort()
            logger.info(f"model_info_dict:{self.model_info_dict}")
            logger.info(f"available_seq_lens:{available_seq_lens}")
            # 寻找合适的模型
            selected_seq_len = None
            for seq_len in available_seq_lens:
                if seq_len >= all_token_nums:
                    selected_seq_len = seq_len
                    break
            # 如果没有找到合适的，选择最大的
            if selected_seq_len is None:
                selected_seq_len = available_seq_lens[-1]
            logger.info(f"selected_seq_len:{selected_seq_len}")
            self.model_info = self.model_info_dict[selected_seq_len]
            vastai_rerank = self.model_info._model.infer(self.model_info.vsx_engine, sentence_combinations) 
            similarity_scores = [ele[0] for ele in vastai_rerank]
            similarity_scores = np.array(similarity_scores)
            logger.info(f"{similarity_scores}")
        elif self._model_spec.type == "normal":
            similarity_scores = self._model.predict(
                sentence_combinations,
                convert_to_numpy=False,
                convert_to_tensor=True,
                **kwargs,
            ).cpu()
            if similarity_scores.dtype == torch.bfloat16:
                similarity_scores = similarity_scores.float()
        else:
            # Related issue: https://github.com/xorbitsai/inference/issues/1775
            similarity_scores = self._model.compute_score(
                sentence_combinations, **kwargs
            )

            if not isinstance(similarity_scores, Sequence):
                similarity_scores = [similarity_scores]
            elif (
                isinstance(similarity_scores, list)
                and len(similarity_scores) > 0
                and isinstance(similarity_scores[0], Sequence)
            ):
                similarity_scores = similarity_scores[0]

        sim_scores_argsort = list(reversed(np.argsort(similarity_scores)))
        if top_n is not None:
            sim_scores_argsort = sim_scores_argsort[:top_n]
        if return_documents:
            docs = [
                DocumentObj(
                    index=int(arg),
                    relevance_score=float(similarity_scores[arg]),
                    document=Document(text=documents[arg]),
                )
                for arg in sim_scores_argsort
            ]
        else:
            docs = [
                DocumentObj(
                    index=int(arg),
                    relevance_score=float(similarity_scores[arg]),
                    document=None,
                )
                for arg in sim_scores_argsort
            ]
        if return_len:
            if self._model_spec.type != "vastai":
                input_len = self._model.model.n_tokens
            else:
                input_len = 512
            # Rerank Model output is just score or documents
            # while return_documents = True
            output_len = input_len

        # api_version, billed_units, warnings
        # is for Cohere API compatibility, set to None
        metadata = {
            "api_version": None,
            "billed_units": None,
            "tokens": (
                RerankTokens(input_tokens=input_len, output_tokens=output_len)
                if return_len
                else None
            ),
            "warnings": None,
        }

        del similarity_scores
        # clear cache if possible
        self._counter += 1
        if self._counter % RERANK_EMPTY_CACHE_COUNT == 0:
            logger.debug("Empty rerank cache.")
            gc.collect()
            empty_cache()

        return Rerank(id=str(uuid.uuid1()), results=docs, meta=metadata)


def create_rerank_model_instance(
    subpool_addr: str,
    devices: List[str],
    model_uid: str,
    model_name: str,
    download_hub: Optional[
        Literal["huggingface", "modelscope", "openmind_hub", "csghub"]
    ] = None,
    model_path: Optional[str] = None,
    **kwargs,
) -> Tuple[RerankModel, RerankModelDescription]:
    from xinference.model.utils import download_from_modelscope
    from xinference.model.rerank import BUILTIN_RERANK_MODELS, MODELSCOPE_RERANK_MODELS
    from xinference.model.rerank.custom import get_user_defined_reranks

    model_spec = None
    for ud_spec in get_user_defined_reranks():
        if ud_spec.model_name == model_name:
            model_spec = ud_spec
            break

    if model_spec is None:
        if download_hub == "huggingface" and model_name in BUILTIN_RERANK_MODELS:
            logger.debug(f"Rerank model {model_name} found in Huggingface.")
            model_spec = BUILTIN_RERANK_MODELS[model_name]
        elif download_hub == "modelscope" and model_name in MODELSCOPE_RERANK_MODELS:
            logger.debug(f"Rerank model {model_name} found in ModelScope.")
            model_spec = MODELSCOPE_RERANK_MODELS[model_name]
        elif download_from_modelscope() and model_name in MODELSCOPE_RERANK_MODELS:
            logger.debug(f"Rerank model {model_name} found in ModelScope.")
            model_spec = MODELSCOPE_RERANK_MODELS[model_name]
        elif model_name in BUILTIN_RERANK_MODELS:
            logger.debug(f"Rerank model {model_name} found in Huggingface.")
            model_spec = BUILTIN_RERANK_MODELS[model_name]
        else:
            raise ValueError(
                f"Rerank model {model_name} not found, available"
                f"Huggingface: {BUILTIN_RERANK_MODELS.keys()}"
                f"ModelScope: {MODELSCOPE_RERANK_MODELS.keys()}"
            )
    if model_path is None:
        if (
            hasattr(model_spec, "model_uri")
            and getattr(model_spec, "model_uri", None) is not None
        ):
            logger.info(f"Model caching from URI: {model_spec.model_uri}")
            if model_spec.model_uri.endswith("mod"):
                model_path = model_spec.model_uri
        else:
            model_path = cache(model_spec)
    use_fp16 = kwargs.pop("use_fp16", False)
    logger.info(f"create_rerank_instance:model_path:{model_path}, device id:{devices[0]}")
    model = RerankModel(
        model_spec, model_uid, model_path, devices[0], use_fp16=use_fp16, model_config=kwargs
    )
    model_description = RerankModelDescription(
        subpool_addr, devices, model_spec, model_path=model_path
    )
    return model, model_description
