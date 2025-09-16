import logging
from importlib import import_module
from xinference.deploy import cmdline
from typing import Dict, List
from .utils import VAPatchesManager, check_package_version
from .device_utils import DeviceType,DEVICE_TO_ENV_NAME,get_available_device,gpu_count
logger = logging.getLogger(__name__)

import sys
import importlib.util
from pathlib import Path
import os
def inject_vastai_core():
    # 1. 获取目标模块路径（xinference.model.embedding）
    try:
        import xinference.model.embedding as embedding_module
        embedding_path = Path(embedding_module.__file__).parent
    except ImportError:
        raise RuntimeError("xinference 未安装！")

    # 2. 指向你的 vastai_core.py（在补丁包内）

    vastai_core_path = os.path.join(Path(__file__).parent, "model/embedding/vastai_core.py")
    # vastai_core_path = Path(__file__).parent/ / "vastai_core.py"

    # 3. 动态加载模块
    spec = importlib.util.spec_from_file_location(
        "xinference.model.embedding.vastai_core",
        vastai_core_path
    )
    vastai_module = importlib.util.module_from_spec(spec)
    sys.modules["xinference.model.embedding.vastai_core"] = vastai_module
    spec.loader.exec_module(vastai_module)

    # 4. 绑定到原模块的命名空间
    setattr(embedding_module, "vastai_core", vastai_module)

# 应用补丁
inject_vastai_core()

import wrapt
@wrapt.decorator
def wrap_init(wrapped, instance, args, kwargs):
    # 先调用原始初始化方法
    result = wrapped(*args, **kwargs)
    # 确保属性存在
    if not hasattr(instance, '_vastai_UID_emb'):
        instance._vastai_UID_emb: Dict[str, List[int]] = {}
        logger.warning("patch _vastai_UID_emb for workerActor")
    if not hasattr(instance, '_vastai_UID_rerank'):
        instance._vastai_UID_rerank: Dict[str, List[int]] = {}
        logger.warning("patch _vastai_UID_rerank for workerActor")
    return result

class XinferPatchManager(VAPatchesManager):
    """Patch Manager"""
    patches_info = {}

    @classmethod
    def check_env(cls):
        return check_package_version("xinference", "1.5.1", "eq")

    @classmethod
    # NOTE（wyl）: convenient for subsequent compatibility with vsx
    def patch_engine(cls) -> None:
        """Patch engine with optional modules.

        Args:
            modules: List of module names to import (default: ["torch_vacc", "vllm_vacc"])
        """
        for module in  ["torch", "torch_vacc", "vllm_vacc", "vaststreamx"]:
            try:
                import_module(module)
            except ImportError as e:
                logger.error(f"Warning: Failed to import {module} ({e})")

    @classmethod
    def patch_device(cls):
        cls.register_patch('xinference.device_utils.DeviceType', DeviceType)
        cls.register_patch('xinference.device_utils.DEVICE_TO_ENV_NAME', DEVICE_TO_ENV_NAME)
        cls.register_patch('xinference.device_utils.get_available_device', get_available_device)
        cls.register_patch('xinference.device_utils.gpu_count', gpu_count)

    @classmethod
    def patch_embed(cls):
        # 应用补丁
        from xinference.core.worker import WorkerActor
        cls.register_patch('xinference.core.worker.WorkerActor.__init__', wrap_init(WorkerActor.__init__))
        from xinference_vacc.core.worker import _create_subpool as _create_subpool_vacc
        cls.register_patch('xinference.core.worker.WorkerActor._create_subpool', _create_subpool_vacc)
        from xinference_vacc.model.utils import is_valid_model_uri as is_valid_model_uri_vacc
        cls.register_patch('xinference.model.utils.is_valid_model_uri', is_valid_model_uri_vacc)
        from xinference_vacc.model.embedding.core import create_embedding_model_instance as create_embedding_model_instance_vacc
        cls.register_patch('xinference.model.embedding.core.create_embedding_model_instance', create_embedding_model_instance_vacc)
        from xinference_vacc.model.rerank.core import create_rerank_model_instance as create_rerank_model_instance_vacc
        cls.register_patch('xinference.model.rerank.core.create_rerank_model_instance', create_rerank_model_instance_vacc)


        
    @classmethod
    def exec_adaptor(cls):
        if cls.check_env():
            logger.warning("Xinference is detected, applying patches for vacc ...")
            cls.patch_engine()
            cls.patch_device()
            cls.patch_embed()
            cls.apply_patches()
##替换类, 包括worker, embedding, rerank
def patch_block_manager():
    from xinference_vacc.model.embedding.core import EmbeddingModel as VaccEmbeddingModel
    import xinference.model.embedding.core
    setattr(xinference.model.embedding.core, "EmbeddingModel", VaccEmbeddingModel)
    logger.warning("patch for vacc Embedding model class")
    from xinference_vacc.model.rerank.core import RerankModel as VaccRerankModel
    import xinference.model.rerank.core
    setattr(xinference.model.rerank.core, "RerankModel", VaccRerankModel)
    logger.warning("patch for vacc Rerank model class")


XinferPatchManager.exec_adaptor()
patch_block_manager()

def set_vacc_visible_devices():
    """
    Only when the current process is launched by 'xinference-worker',
    """
    # 获取脚本名，判断是否为 xinference-worker
    script_name = os.path.basename(sys.argv[0]).lower()
    if script_name not in {"xinference-worker"}:
        return  # 非 worker，不处理

    # 获取用户设置的 VACC_VISIBLE_DEVICES
    vacc_visible = os.getenv("VACC_VISIBLE_DEVICES")
    if not vacc_visible:
        # 如果没设置，可以忽略，或打个日志
        print("Warning: VACC_VISIBLE_DEVICES is not set. Using all available VACC devices by default?")
        import glob
        vacc_devices = glob.glob('/dev/vacc*')
        count = len(vacc_devices)
        vacc_visible = ",".join(str(i) for i in range(count))
    # 输出确认信息
    print(f"VACC_VISIBLE_DEVICES: {vacc_visible}")

# 执行设置
set_vacc_visible_devices()

os.environ["XINFERENCE_SSE_PING_ATTEMPTS_SECONDS"] = "864000"
os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "864000" 
# NOTE: init patch, if not, will not auto path this script
def cli():
    cmdline.cli()
def local():
    cmdline.local()
def supervisor():
    cmdline.supervisor()
def worker():
    cmdline.worker()
