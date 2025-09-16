import os
from queue import Queue
from threading import Event, Thread
from typing import Dict, Iterable, List, Union

import numpy as np
from tqdm.autonotebook import trange
import torch

try:
    import vaststreamx
except ImportError:
    error_message = "Failed to import module 'vaststreamx'"
    installation_guide = [
        "Please make sure 'vaststreamx' is installed.\n",
    ]
    raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
# import vaststreamx as vaststreamx

class EmbeddingX:
    def __init__(
        self,
        model_prefix_path: Union[str, Dict[str, str]],
        device_id: int = 0,
        batch_size: int = 1,
        is_async_infer: bool = False,
        model_output_op_name: str = "",
    ):
        self.device_id = device_id
        self.model_output_op_name = model_output_op_name
        self.input_id = 0 

        self.attr = vaststreamx.AttrKey 
        vaststreamx.set_device(self.device_id)
        # 构建model，模型三件套目录
        model_path = model_prefix_path 
        self.model = vaststreamx.Model(model_path, batch_size) 
        # 输入预处理op
        self.embedding_op = vaststreamx.Operator(vaststreamx.OpType.BERT_EMBEDDING_OP) 
        # 有以上op时无法载通过vaststreamx.Operator加载vdsp算子
        # self.fusion_op = vaststreamx.Operator.load_ops_from_json_file(vdsp_params_info)[0] 
        # 构建graph
        self.graph = vaststreamx.Graph(do_copy=False)
        self.model_op = vaststreamx.ModelOperator(self.model)
        self.graph.add_operators(self.embedding_op, self.model_op) 

        # 构建stream
        self.infer_stream = vaststreamx.Stream(self.graph, vaststreamx.StreamBalanceMode.ONCE)
        if is_async_infer and len(self.model_output_op_name) > 0:
            self.infer_stream.register_operator_output(self.model_output_op_name, self.model_op)
        else:
            self.infer_stream.register_operator_output(self.model_op) 

        # # 预处理算子输出
        self.infer_stream.build() 

        self.current_id = -1
        self.input_dict = {}
        self.event_dict = {}
        self.result_dict = {}
        # 异步
        self.consumer = Thread(target=self.async_receive_infer)
        self.consumer.start()

    def async_receive_infer(self):
        while 1:
            try:
                result = None
                if len(self.model_output_op_name) > 0:
                    result = self.infer_stream.get_operator_output(self.model_output_op_name)
                else:
                    result = self.infer_stream.get_operator_output(self.model_op)
                if result is not None:
                    # pre_process_tensor = self.infer_stream.get_operator_output(self.preprocess_name)
                    # 输出顺序和输入一致
                    self.current_id += 1
                    (input_id,) = self.input_dict[self.current_id]
                    model_output_list = [
                        [vaststreamx.as_numpy(out).astype(np.float32) for out in result[0]]
                    ]
                    self.result_dict[input_id] = []
                    self.post_processing(input_id, model_output_list)
                    self.event_dict[input_id].set()
            except ValueError as e:
                error_message = str(e)
                break

    def post_processing(self, input_id, stream_output_list):
        output_data = stream_output_list[0][0]

        self.result_dict[input_id].append(
            {
                "output": output_data,
            }
        )

    def _run(self, vaststreamx_tensors):
        input_id = self.input_id
        self.input_dict[input_id] = (input_id,)
        self.event_dict[input_id] = Event()

        self.infer_stream.run_async([vaststreamx_tensors])
        self.input_id += 1
        return input_id

    def run_batch(self, datasets: Iterable[List[np.ndarray]]):
        queue = Queue(20)

        def input_thread():
            for data in datasets:
                input_id = self._run(data)
                queue.put(input_id)
            queue.put(None)

        thread = Thread(target=input_thread)
        thread.start()
        while True:
            input_id = queue.get()
            if input_id is None:
                break
            self.event_dict[input_id].wait()
            result = self.result_dict.pop(input_id)
            del self.event_dict[input_id]
            yield result

    def save(self, out, save_dir, name):
        outputs = {}
        outputs = {f'output_{i}': o['output'] for i, o in enumerate(out)}
        np.savez(os.path.join(save_dir, name), **outputs)

    def finish(self):
        self.infer_stream.close_input()
        self.infer_stream.wait_until_done()
        self.consumer.join()
        print("************/n vaststreamx engine del /n************")

    def dummy_datasets(self):

        def dataset_loader():
            npz_datalist = range(300)
            for index, data_path in enumerate(npz_datalist):
                inputs = [np.ones((1, 512)) for _ in range(6)]
                vaststreamx_tensors = [
                    vaststreamx.from_numpy(np.array(input, dtype=np.int32), self.device_id)
                    for input in inputs
                ]

                yield vaststreamx_tensors

        return dataset_loader

    def __call__(self, inputs):
        outputs = self.infer_stream.run_sync(inputs)
        return np.array([vaststreamx.as_numpy(outputs[i][0]) for i in range(len(outputs))])


class VastaiEmbedding:
    def __init__(
        self,
        model_prefix_path: Union[str, Dict[str, str]],
        torch_model_or_tokenizer: str,
        device_id: int = 0,
        batch_size: int = 4,
        max_seqlen: int = 2048,
    ):

        self.max_seqlen = max_seqlen
        self.torch_model_or_tokenizer = torch_model_or_tokenizer
        self.batch_size = batch_size
        self.device_id = device_id
        self.model_prefix_path = model_prefix_path

        try:
            from transformers import AutoTokenizer
        except ImportError:
            error_message = "Failed to import'AutoTokenizer' from transformers"
            installation_guide = [
                "Please make sure 'transformers' is installed.\n",
            ]
            raise ImportError(f"{error_message}\n\n{''.join(installation_guide)}")
        # init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.torch_model_or_tokenizer, trust_remote_code=True
        )

    def init_vaststreamx(self):
        # init vaststreamx
        vaststreamx_engine = EmbeddingX(
            model_prefix_path=self.model_prefix_path,
            device_id=self.device_id,
            batch_size=self.batch_size,
        )
        print("vaststreamx model init done.")
        return vaststreamx_engine

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings

    def infer(self, engine, sentences: Union[str, List[str]]) -> List:

        all_embeddings = []

        if isinstance(sentences, str):
            sentences = [sentences]
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]
        for start_index in trange(0, len(sentences), self.batch_size, desc="Batches"):
            sentences_batch = sentences_sorted[start_index : start_index + self.batch_size]
            features = self.tokenizer(
                sentences_batch,
                # vacc 无论是True还是"max_length"，都是按照定长输出的， #max_length的长度 输出的
                # torch 可以根据True还是"max_length"配置padding，按照实际encode中最长的输出
                padding="max_length",
                truncation=True,
                max_length=self.max_seqlen,
                return_tensors="np",
            )
            if isinstance(engine, EmbeddingX):
                token_embeddings = self.run_vaststreamx(engine, features)

            token_embeddings = self.post_precess(token_embeddings)
            all_embeddings.extend(token_embeddings)
        return np.array(all_embeddings)

    def run_vaststreamx(self, vaststreamx_engine, features: Dict[str, np.ndarray]):
        # add for vaststreamx
        features['token_type_ids'] = np.zeros(features['input_ids'].shape, dtype=np.int32)
        # default order array
        vaststreamx_inputs = [
            features['input_ids'],
            features['attention_mask'],
            features['token_type_ids'],
            features['attention_mask'],
            features['attention_mask'],
            features['attention_mask'],
        ]

        # split to batches
        vaststreamx_inputs = np.concatenate(vaststreamx_inputs, axis=0)
        vaststreamx_inputs = np.split(vaststreamx_inputs, vaststreamx_inputs.shape[0], axis=0)
        vaststreamx_batches = []
        for i in range(len(vaststreamx_inputs) // 6):
            vaststreamx_batch = []
            for inp in vaststreamx_inputs[i :: len(vaststreamx_inputs) // 6]:
                vaststreamx_batch.append(
                    vaststreamx.from_numpy(
                        np.array(inp, dtype=np.int32),
                        self.device_id if hasattr(self, 'device_id') else 0,
                    )
                )
            vaststreamx_batches.append(vaststreamx_batch)
        
        ori_vaststreamx_output = vaststreamx_engine(vaststreamx_batches)
        return ori_vaststreamx_output #vaststreamx_engine(vaststreamx_batches)

    def post_precess(self, outputs: np.ndarray) -> List:
        embeddings = outputs[:, 0]
        embeddings = torch.from_numpy(embeddings)
        embeddings = torch.sigmoid(embeddings.double())
        return embeddings.detach().cpu().numpy()


class VastaiReranker(VastaiEmbedding):
    def infer(self, engine, sentences: List[List[str]]) -> List:

        all_embeddings = []

        for start_index in trange(0, len(sentences), self.batch_size, desc="Batches"):
            sentences_batch = sentences[start_index : start_index + self.batch_size]
            features = self.tokenizer(
                sentences_batch,
                padding="max_length",
                truncation=True,
                return_tensors="np",
            )
            if isinstance(engine, EmbeddingX):
                token_embeddings = self.run_vaststreamx(engine, features)
                token_embeddings = self.post_precess(token_embeddings)
            all_embeddings.extend(token_embeddings)
        return np.array(all_embeddings)
