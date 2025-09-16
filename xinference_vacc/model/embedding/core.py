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
import logging
import os
from collections import defaultdict
from typing import Dict, List, Literal, Optional, Tuple, Union, no_type_check

import numpy as np
import torch

from xinference.model.embedding.core import EmbeddingModelSpec, EmbeddingModelDescription, match_embedding
from xinference.types import EmbeddingData, EmbeddingUsage, Embedding
from xinference.device_utils import empty_cache
import logging
import json
from transformers import AutoTokenizer
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)
MODEL_NAME_TO_REVISION: Dict[str, List[str]] = defaultdict(list)
EMBEDDING_MODEL_DESCRIPTIONS: Dict[str, List[Dict]] = defaultdict(list)
EMBEDDING_EMPTY_CACHE_COUNT = int(
    os.getenv("XINFERENCE_EMBEDDING_EMPTY_CACHE_COUNT", "10")
)
EMBEDDING_EMPTY_CACHE_TOKENS = int(
    os.getenv("XINFERENCE_EMBEDDING_EMPTY_CACHE_TOKENS", "8192")
)
assert EMBEDDING_EMPTY_CACHE_COUNT > 0
assert EMBEDDING_EMPTY_CACHE_TOKENS > 0
class EmbeddingModel:
    class ModelInfo:
        def __init__(self, model_path: str, batch_size: int, max_seqlen: int):
            self.model_path = model_path
            self.batch_size = batch_size
            self.max_seqlen = max_seqlen
            self._model = None  # 初始化为 None
            self.vsx_engine = None 
    def __init__(
        self,
        model_uid: str,
        model_path: str,
        model_spec: EmbeddingModelSpec,
        device: Optional[str] = None,
        **kwargs,
    ):
        self._model_uid = model_uid
        self._model_path = model_path
        self._device = device
        self._model = None
        self._counter = 0
        self._model_spec = model_spec
        self._kwargs = kwargs
        self.vastai_emb = False
        self.vastai_tokenizer = None
        logger.info(f"embedding model_path:{model_path}")
        # 存储所有 ModelInfo 实例的列表, 状态字典
        self.modelInfo_list = []
        self.model_info_dict = {}
        ##加载三件套
        if self._model_path.endswith("mod"):
            self.vastai_emb = True
            self._model_tokenizer = self._model_path[:-3] + "512/tokenizer"
            logger.info(f"Tokenizer path: {self._model_tokenizer}")
            # 加载一次 tokenizer
            try:
                self.vastai_tokenizer = AutoTokenizer.from_pretrained(self._model_tokenizer)
            except Exception as e:
                logger.error(f"Failed to load tokenizer: {e}")
                self.vastai_tokenizer = None
        ##如果有子目录 512/1024/2048/4096/8192
        self.root_dir = self._model_path[:-3]
        subdirs_list = self._find_model_subdirs()
        for i in subdirs_list:
            # 如果有子目录
            subdir_modpath = os.path.join(self.root_dir, i, "mod")
            logger.info(f"subdirectory mod path: {subdir_modpath}")
            model_info = self._initialize_vastai_model(subdir_modpath)
            self.modelInfo_list.append(model_info)
        logger.info(f"目前modelInfo list 长度: {len(self.modelInfo_list)}")
    
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
                logger.info(f"Read embedding config from {vacc_config_path}")
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

    def load(self):
        if self.vastai_emb:
            logger.info("now loading vastai embedding")
            self.vastai_embedding_id = int(self._device)
            logger.info(f"self.vastai_embedding_id:{self.vastai_embedding_id}")
            from .vastai_core import VastaiEmbedding
            for model_info in self.modelInfo_list:
                model_info._model = VastaiEmbedding(
                    model_info.model_path,
                    self._model_tokenizer,
                    device_id=self.vastai_embedding_id,
                    batch_size=model_info.batch_size,
                    max_seqlen=model_info.max_seqlen,
                )
                logger.info(f"embedding model info:{model_info.model_path}, {model_info.batch_size}, {model_info.max_seqlen}")
                model_info.vsx_engine = model_info._model.init_vaststreamx()
                #{512：model_info, 1024:model_info}
                self.model_info_dict[model_info.max_seqlen] = model_info
            logger.info(f"finish load vastai embedding with id:{self.vastai_embedding_id}")
            return
        try:
            import sentence_transformers
            from sentence_transformers import SentenceTransformer

            if sentence_transformers.__version__ < "3.1.0":
                raise ValueError(
                    "The sentence_transformers version must be greater than 3.1.0. "
                    "Please upgrade your version via `pip install -U sentence_transformers` or refer to "
                    "https://github.com/UKPLab/sentence-transformers"
                )
        except ImportError:
            error_message = "Failed to import module 'SentenceTransformer'"
            installation_guide = [
                "Please make sure 'sentence-transformers' is installed. ",
                "You can install it by `pip install sentence-transformers`\n",
            ]
            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

        class XSentenceTransformer(SentenceTransformer):
            def to(self, *args, **kwargs):
                pass

        torch_dtype = None
        if torch_dtype_str := self._kwargs.get("torch_dtype"):
            try:
                torch_dtype = getattr(torch, torch_dtype_str)
                if torch_dtype not in [
                    torch.float16,
                    torch.float32,
                    torch.bfloat16,
                ]:
                    logger.warning(
                        f"Load embedding model with unsupported torch dtype :  {torch_dtype_str}. Using default torch dtype: fp32."
                    )
                    torch_dtype = torch.float32
            except AttributeError:
                logger.warning(
                    f"Load embedding model with  unknown torch dtype '{torch_dtype_str}'. Using default torch dtype: fp32."
                )
                torch_dtype = torch.float32

        if (
            "gte" in self._model_spec.model_name.lower()
            and "qwen2" in self._model_spec.model_name.lower()
        ):
            model_kwargs = {"device_map": "auto"}
            if torch_dtype:
                model_kwargs["torch_dtype"] = torch_dtype
            self._model = XSentenceTransformer(
                self._model_path,
                device=self._device,
                model_kwargs=model_kwargs,
            )
        elif (
            self._kwargs.get("hybrid_mode")
            and "m3" in self._model_spec.model_name.lower()
        ):
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError:
                error_message = "Failed to import module 'BGEM3FlagModel'"
                installation_guide = [
                    "Please make sure 'FlagEmbedding' is installed. ",
                    "You can install it by `pip install FlagEmbedding`\n",
                ]
                raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")

            if torch_dtype and torch_dtype == torch.float16:
                model_kwargs = {"use_fp16": True}
            else:
                model_kwargs = {}
            self._model = BGEM3FlagModel(
                self._model_path,
                device=self._device,
                **model_kwargs,
            )
        else:
            model_kwargs = {"torch_dtype": torch_dtype} if torch_dtype else None
            self._model = SentenceTransformer(
                self._model_path,
                device=self._device,
                model_kwargs=model_kwargs,
                trust_remote_code=True,
            )

    def _fix_langchain_openai_inputs(
        self, sentences: Union[str, List[str], Dict[str, str], List[Dict[str, str]]]
    ):
        # Check if sentences is a two-dimensional list of integers
        if (
            isinstance(sentences, list)
            and len(sentences) > 0
            and isinstance(sentences[0], list)
            and len(sentences[0]) > 0
            and isinstance(sentences[0][0], int)
        ):
            # List[List[int]] stands for encoded inputs
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            lines_decoded = []

            for line in sentences:
                try:
                    # Decode each token into bytes, then join them into a complete string
                    output = b"".join(
                        enc.decode_single_token_bytes(token) for token in line
                    )
                    # Convert the byte sequence into a UTF-8 encoded string
                    decoded_line = output.decode("utf-8")
                    lines_decoded.append(decoded_line)
                except (ValueError, TypeError, UnicodeDecodeError) as e:
                    raise ValidationError([ErrorWrapper(e, loc=ROOT_KEY)], self)

            # Update sentences to be the list of decoded strings
            if len(lines_decoded) == 1:
                sentences = lines_decoded[0]
            else:
                sentences = lines_decoded
        return sentences

    def create_embedding(
        self,
        sentences: Union[str, List[str]],
        **kwargs,
    ):
        sentences = self._fix_langchain_openai_inputs(sentences)
        model_uid = kwargs.pop("model_uid", None)
        from sentence_transformers import SentenceTransformer

        kwargs.setdefault("normalize_embeddings", True)

        try:
            from FlagEmbedding import BGEM3FlagModel

            @no_type_check
            def _encode_bgem3(
                model: Union[SentenceTransformer, BGEM3FlagModel],
                sentences: Union[str, List[str]],
                batch_size: int = 32,
                show_progress_bar: bool = None,
                output_value: str = "sparse_embedding",
                convert_to_numpy: bool = True,
                convert_to_tensor: bool = False,
                device: str = None,
                normalize_embeddings: bool = False,
                **kwargs,
            ):
                """
                Computes sentence embeddings with bge-m3 model
                Nothing special here, just replace sentence-transformer with FlagEmbedding
                TODO: think about how to solve the redundant code of encode method in the future

                :param sentences: the sentences to embed
                :param batch_size: the batch size used for the computation
                :param show_progress_bar: Output a progress bar when encode sentences
                :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
                :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
                :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
                :param device: Which torch.device to use for the computation
                :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

                :return:
                    By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
                """
                import torch
                from tqdm.autonotebook import trange

                if show_progress_bar is None:
                    show_progress_bar = (
                        logger.getEffectiveLevel() == logging.INFO
                        or logger.getEffectiveLevel() == logging.DEBUG
                    )

                if convert_to_tensor:
                    convert_to_numpy = False

                if output_value != "sparse_embedding":
                    convert_to_tensor = False
                    convert_to_numpy = False

                input_was_string = False
                if isinstance(sentences, str) or not hasattr(
                    sentences, "__len__"
                ):  # Cast an individual sentence to a list with length 1
                    sentences = [sentences]
                    input_was_string = True

                if device is None:
                    # Same as SentenceTransformer.py
                    from sentence_transformers.util import get_device_name

                    device = get_device_name()
                    logger.info(f"Use pytorch device_name: {device}")

                all_embeddings = []
                all_token_nums = 0

                # The original code does not support other inference engines
                def _text_length(text):
                    if isinstance(text, dict):  # {key: value} case
                        return len(next(iter(text.values())))
                    elif not hasattr(text, "__len__"):  # Object has no len() method
                        return 1
                    elif len(text) == 0 or isinstance(
                        text[0], int
                    ):  # Empty string or list of ints
                        return len(text)
                    else:
                        return sum(
                            [len(t) for t in text]
                        )  # Sum of length of individual strings

                length_sorted_idx = np.argsort(
                    [-_text_length(sen) for sen in sentences]
                )
                sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

                for start_index in trange(
                    0,
                    len(sentences),
                    batch_size,
                    desc="Batches",
                    disable=not show_progress_bar,
                ):
                    sentences_batch = sentences_sorted[
                        start_index : start_index + batch_size
                    ]

                    with torch.no_grad():
                        out_features = model.encode(sentences_batch, **kwargs)

                        if output_value == "token_embeddings":
                            embeddings = []
                            for token_emb, attention in zip(
                                out_features[output_value],
                                out_features["attention_mask"],
                            ):
                                last_mask_id = len(attention) - 1
                                while (
                                    last_mask_id > 0
                                    and attention[last_mask_id].item() == 0
                                ):
                                    last_mask_id -= 1

                                embeddings.append(token_emb[0 : last_mask_id + 1])
                        elif output_value is None:  # Return all outputs
                            embeddings = []
                            for sent_idx in range(
                                len(out_features["sentence_embedding"])
                            ):
                                row = {
                                    name: out_features[name][sent_idx]
                                    for name in out_features
                                }
                                embeddings.append(row)
                        # for sparse embedding
                        else:
                            if kwargs.get("return_sparse"):
                                embeddings = out_features["lexical_weights"]
                            else:
                                embeddings = out_features["dense_vecs"]

                            if convert_to_numpy:
                                embeddings = embeddings.cpu()

                        all_embeddings.extend(embeddings)

                all_embeddings = [
                    all_embeddings[idx] for idx in np.argsort(length_sorted_idx)
                ]

                if convert_to_tensor:
                    if len(all_embeddings):
                        all_embeddings = torch.stack(all_embeddings)
                    else:
                        all_embeddings = torch.Tensor()
                elif convert_to_numpy:
                    all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

                if input_was_string:
                    all_embeddings = all_embeddings[0]

                return all_embeddings, all_token_nums

        except ImportError:
            _encode_bgem3 = None

        # copied from sentence-transformers, and modify it to return tokens num
        @no_type_check
        def encode(
            model: SentenceTransformer,
            sentences: Union[str, List[str]],
            prompt_name: Optional[str] = None,
            prompt: Optional[str] = None,
            batch_size: int = 32,
            show_progress_bar: bool = None,
            output_value: str = "sentence_embedding",
            convert_to_numpy: bool = True,
            convert_to_tensor: bool = False,
            device: str = None,
            normalize_embeddings: bool = False,
            **kwargs,
        ):
            """
            Computes sentence embeddings

            :param sentences: the sentences to embed
            :param batch_size: the batch size used for the computation
            :param show_progress_bar: Output a progress bar when encode sentences
            :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
            :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
            :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
            :param device: Which torch.device to use for the computation
            :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

            :return:
               By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
            """
            import torch
            from sentence_transformers.util import batch_to_device
            from tqdm.autonotebook import trange

            model.eval()
            if show_progress_bar is None:
                show_progress_bar = (
                    logger.getEffectiveLevel() == logging.INFO
                    or logger.getEffectiveLevel() == logging.DEBUG
                )

            if convert_to_tensor:
                convert_to_numpy = False

            if output_value != "sentence_embedding":
                convert_to_tensor = False
                convert_to_numpy = False

            input_was_string = False
            if isinstance(sentences, str) or not hasattr(
                sentences, "__len__"
            ):  # Cast an individual sentence to a list with length 1
                sentences = [sentences]
                input_was_string = True

            if prompt is None:
                if prompt_name is not None:
                    try:
                        prompt = model.prompts[prompt_name]
                    except KeyError:
                        raise ValueError(
                            f"Prompt name '{prompt_name}' not found in the configured prompts dictionary with keys {list(model.prompts.keys())!r}."
                        )
                elif model.default_prompt_name is not None:
                    prompt = model.prompts.get(model.default_prompt_name, None)
            else:
                if prompt_name is not None:
                    logger.warning(
                        "Encode with either a `prompt`, a `prompt_name`, or neither, but not both. "
                        "Ignoring the `prompt_name` in favor of `prompt`."
                    )

            extra_features = {}
            if prompt is not None:
                sentences = [prompt + sentence for sentence in sentences]

                # Some models (e.g. INSTRUCTOR, GRIT) require removing the prompt before pooling
                # Tracking the prompt length allow us to remove the prompt during pooling
                tokenized_prompt = model.tokenize([prompt])
                if "input_ids" in tokenized_prompt:
                    extra_features["prompt_length"] = (
                        tokenized_prompt["input_ids"].shape[-1] - 1
                    )

            if device is None:
                device = model._target_device

            if (
                "gte" in self._model_spec.model_name.lower()
                and "qwen2" in self._model_spec.model_name.lower()
            ):
                model.to(device)

            all_embeddings = []
            all_token_nums = 0
            length_sorted_idx = np.argsort(
                [-model._text_length(sen) for sen in sentences]
            )
            sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

            for start_index in trange(
                0,
                len(sentences),
                batch_size,
                desc="Batches",
                disable=not show_progress_bar,
            ):
                sentences_batch = sentences_sorted[
                    start_index : start_index + batch_size
                ]
                features = model.tokenize(sentences_batch)
                features = batch_to_device(features, device)
                features.update(extra_features)
                # when batching, the attention mask 1 means there is a token
                # thus we just sum up it to get the total number of tokens
                if "clip" in self._model_spec.model_name.lower():
                    if "input_ids" in features and hasattr(
                        features["input_ids"], "numel"
                    ):
                        all_token_nums += features["input_ids"].numel()
                    if "pixel_values" in features and hasattr(
                        features["pixel_values"], "numel"
                    ):
                        all_token_nums += features["pixel_values"].numel()
                else:
                    all_token_nums += features["attention_mask"].sum().item()

                with torch.no_grad():
                    out_features = model.forward(features, **kwargs)

                    if output_value == "token_embeddings":
                        embeddings = []
                        for token_emb, attention in zip(
                            out_features[output_value], out_features["attention_mask"]
                        ):
                            last_mask_id = len(attention) - 1
                            while (
                                last_mask_id > 0 and attention[last_mask_id].item() == 0
                            ):
                                last_mask_id -= 1

                            embeddings.append(token_emb[0 : last_mask_id + 1])
                    elif output_value is None:  # Return all outputs
                        embeddings = []
                        for sent_idx in range(len(out_features["sentence_embedding"])):
                            row = {
                                name: out_features[name][sent_idx]
                                for name in out_features
                            }
                            embeddings.append(row)
                    else:  # Sentence embeddings
                        embeddings = out_features[output_value]
                        embeddings = embeddings.detach()
                        if normalize_embeddings:
                            embeddings = torch.nn.functional.normalize(
                                embeddings, p=2, dim=1
                            )

                        # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                        if convert_to_numpy:
                            embeddings = embeddings.cpu()

                    all_embeddings.extend(embeddings)

            all_embeddings = [
                all_embeddings[idx] for idx in np.argsort(length_sorted_idx)
            ]

            if convert_to_tensor:
                if len(all_embeddings):
                    all_embeddings = torch.stack(all_embeddings)
                else:
                    all_embeddings = torch.Tensor()
            elif convert_to_numpy:
                all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

            if input_was_string:
                all_embeddings = all_embeddings[0]

            return all_embeddings, all_token_nums

        is_bge_m3_flag_model = (
            self._kwargs.get("hybrid_mode")
            and "m3" in self._model_spec.model_name.lower()
        )
        all_token_nums = 0
        if self.vastai_emb:
            sentences_list = [sentences] if type(sentences) == str else sentences
            all_token_nums = sum(len(self.vastai_tokenizer.tokenize(s)) for s in sentences_list)
            logger.info(f"all_token_nums:{all_token_nums}")
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
            all_embeddings = self.model_info._model.infer(self.model_info.vsx_engine, sentences)
        else: ##其他非vastai embed
            if (
                "gte" in self._model_spec.model_name.lower()
                and "qwen2" in self._model_spec.model_name.lower()
            ):
                all_embeddings, all_token_nums = encode(
                    self._model,
                    sentences,
                    prompt_name="query",
                    convert_to_numpy=False,
                    **kwargs,
                )
            elif is_bge_m3_flag_model:
                assert _encode_bgem3 is not None
                all_embeddings, all_token_nums = _encode_bgem3(
                    self._model, sentences, convert_to_numpy=False, **kwargs
                )
            elif "clip" in self._model_spec.model_name.lower():
                import base64
                import re
                from io import BytesIO

                from PIL import Image

                def base64_to_image(base64_str: str) -> Image.Image:
                    # base64_data = re.sub("^data:image/.+;base64,", "", base64_str)
                    base64_data = base64_str.split(",", 1)[1]
                    byte_data = base64.b64decode(base64_data)
                    image_data = BytesIO(byte_data)
                    img = Image.open(image_data)
                    return img

                objs: list[dict[str, str]] = []
                for item in sentences:
                    if isinstance(item, dict):
                        if item.get("text") is not None:
                            objs.append(item["text"])
                        elif item.get("image") is not None:
                            if re.match(r"^data:image/.+;base64,", item["image"]):
                                image = base64_to_image(item["image"])
                                objs.append(image)
                            else:
                                objs.append(item["image"])
                        else:
                            logger.error("Please check the input data.")
                all_embeddings, all_token_nums = encode(
                    self._model,
                    objs,
                    convert_to_numpy=False,
                    **kwargs,
                )
            else:
                pass
        if isinstance(sentences, str):
            all_embeddings = [all_embeddings]
        embedding_list = []
        for index, data in enumerate(all_embeddings):
            if kwargs.get("return_sparse") and is_bge_m3_flag_model:
                embedding_list.append(
                    EmbeddingData(
                        index=index,
                        object="embedding",
                        embedding={k: float(v) for k, v in data.items()},
                    )
                )
            else:
                embedding_list.append(
                    EmbeddingData(
                        index=index, object="embedding", embedding=data.tolist()
                    )
                )
        usage = EmbeddingUsage(
            prompt_tokens=all_token_nums, total_tokens=all_token_nums
        )
        result = Embedding(
            object=(
                "list"  # type: ignore
                if not is_bge_m3_flag_model and not kwargs.get("return_sparse")
                else "dict"
            ),
            model=model_uid,  # type: ignore
            model_replica=self._model_uid,
            data=embedding_list,
            usage=usage,
        )

        # clean cache if possible
        self._counter += 1
        if (
            self._counter % EMBEDDING_EMPTY_CACHE_COUNT == 0
            or all_token_nums >= EMBEDDING_EMPTY_CACHE_TOKENS
        ):
            logger.debug(
                "Empty embedding cache, calling count %s, all_token_nums %s",
                self._counter,
                all_token_nums,
            )
            gc.collect()
            empty_cache()

        return result

    def convert_ids_to_tokens(
        self,
        batch_token_ids: Union[List[Union[int, str]], List[List[Union[int, str]]]],
        **kwargs,
    ) -> Union[List[str]]:
        batch_decoded_texts: List[str] = []

        assert self._model is not None

        if isinstance(batch_token_ids, (int, str)):
            return self._model.tokenizer.convert_ids_to_tokens(
                [int(str(batch_token_ids))]
            )[0]

        # check if it's a nested list
        if (
            isinstance(batch_token_ids, list)
            and batch_token_ids
            and isinstance(batch_token_ids[0], list)
        ):
            for token_ids in batch_token_ids:
                token_ids = [int(token_id) for token_id in token_ids]
                batch_decoded_texts.append(
                    self._model.tokenizer.convert_ids_to_tokens(token_ids)
                )
        else:
            batch_token_ids = [int(token_id) for token_id in batch_token_ids]
            batch_decoded_texts = self._model.tokenizer.convert_ids_to_tokens(
                batch_token_ids
            )
        return batch_decoded_texts

def create_embedding_model_instance(
    subpool_addr: str,
    devices: List[str],
    model_uid: str,
    model_name: str,
    download_hub: Optional[
        Literal["huggingface", "modelscope", "openmind_hub", "csghub"]
    ] = None,
    model_path: Optional[str] = None,
    **kwargs,
) -> Tuple[EmbeddingModel, EmbeddingModelDescription]:
    logger.info(f"patch----------model_name:{model_name}")
    model_spec = match_embedding(model_name, download_hub)
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
    logger.info(f"create_embedding_model_instance model_path:{model_path}, device id {devices[0]}")
    model = EmbeddingModel(model_uid, model_path, model_spec, devices[0], **kwargs)
    model_description = EmbeddingModelDescription(
        subpool_addr, devices, model_spec, model_path=model_path
    )
    return model, model_description
