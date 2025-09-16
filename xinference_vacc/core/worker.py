from typing import Any, Dict, List, Optional, Set, Tuple, Union
from xinference_vacc.device_utils import get_available_device, gpu_count
import os
import platform
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
logger.addHandler(handler)

async def _create_subpool(
    self,
    model_uid: str,
    model_type: Optional[str] = None,
    n_gpu: Optional[Union[int, str]] = "auto",
    gpu_idx: Optional[List[int]] = None,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[str, List[str]]:
    last_hyphen_index = model_uid.rfind('-')
    model_name = model_uid[:last_hyphen_index] if last_hyphen_index != -1 else model_uid
    logger.info(f"model_name:{model_name}")
    env = {} if env is None else env
    devices = []
    env_name = get_available_device() or "CUDA_VISIBLE_DEVICES"
    logger.info(f"patch _create_subpool, env_name:{env_name}")
    if gpu_idx is None:
        if isinstance(n_gpu, int) or (n_gpu == "auto" and gpu_count() > 0):
            # Currently, n_gpu=auto means using 1 GPU
            gpu_cnt = n_gpu if isinstance(n_gpu, int) else 1
            devices = (
                [await self.allocate_devices_for_embedding(model_uid)]
                if model_type in ["embedding", "rerank"]
                else self.allocate_devices(model_uid=model_uid, n_gpu=gpu_cnt)
            )
            ## 添加一个case, GPU 和vastai emb/rerank id 对应的，
            ## emb[model_name] = [1,2,4] ##表示这个worker embedding name 下面有die1,2,4 用
            ## 首先，保证有这个环境变量并且模型类型是对应的, 是单die 的
            if model_type == "embedding" and os.getenv("VACC_VISIBLE_DEVICES", None) != None:
                logger.info("now vastai embedding")
                vastai_emb_devices = os.environ.get("VACC_VISIBLE_DEVICES", None).split(',')
                die_count = len(vastai_emb_devices)
                embedding_device_ids_list = []
                for idx in range(die_count):
                    embedding_device_ids_list.append(int(idx))
                logger.info(f"{embedding_device_ids_list} - embedding_device_ids_list")
                used_list = self._vastai_UID_emb.get(model_name, [])
                logger.info(f"{used_list} - embedding used list")
                for idx in embedding_device_ids_list:
                    if idx not in used_list:
                        used_list.append(idx)
                        self._vastai_UID_emb[model_name] = used_list
                        devices = [idx]
                        logger.info(f"{self._vastai_UID_emb[model_name]}, embedding current use device")
                        break
            if model_type == "rerank" and os.getenv("VACC_VISIBLE_DEVICES", None) != None:
                logger.info("now vastai rerank")
                vastai_rerank_devices = os.environ.get("VACC_VISIBLE_DEVICES", None).split(',')
                die_count = len(vastai_rerank_devices)
                rerank_device_ids_list = []
                for idx in range(die_count):
                    rerank_device_ids_list.append(int(idx))
                logger.info(f"rerank_device_ids_list, {rerank_device_ids_list}")
                used_list = self._vastai_UID_rerank.get(model_name, [])
                logger.info(f"rerank used_list: {used_list}")
                for idx in rerank_device_ids_list:
                    if idx not in used_list:
                        used_list.append(idx)
                        self._vastai_UID_rerank[model_name] = used_list
                        devices = [idx]
                        logger.info(f"rerank current use device:{self._vastai_UID_rerank[model_name]}")
                        break
            env[env_name] = ",".join([str(dev) for dev in devices])
            logger.info(f"patch _create_subpool, llm devices:{devices}")
            logger.debug(f"GPU selected: {devices} for model {model_uid}")
        if n_gpu is None:
            env[env_name] = "-1"
            logger.debug(f"GPU disabled for model {model_uid}")
    else:
        assert isinstance(gpu_idx, list)
        devices = await self.allocate_devices_with_gpu_idx(
            model_uid, model_type, gpu_idx  # type: ignore
        )
        env[env_name] = ",".join([str(dev) for dev in devices])

    if os.name != "nt" and platform.system() != "Darwin":
        # Linux
        start_method = "forkserver"
    else:
        # Windows and macOS
        start_method = "spawn"
    subpool_address = await self._main_pool.append_sub_pool(
        env=env, start_method=start_method
    )
    return subpool_address, [str(dev) for dev in devices]