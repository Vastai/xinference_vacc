import os
from typing import Any, Callable, Dict, Optional, Tuple, Union

def is_valid_model_uri(model_uri: Optional[str]) -> bool:
    if not model_uri:
        return False
    ##单独处理mod 结尾的情况
    if model_uri.endswith("mod"):
        root_dir = model_uri[:-3]
        subdirs_list = []
        if os.path.isdir(root_dir):
            for item in os.listdir(root_dir):
                item_path = os.path.join(root_dir, item)
                if os.path.isdir(item_path) and item.isdigit():
                    subdirs_list.append(item)
        # 按数字大小排序
        subdirs_list.sort(key=int)
        for i in subdirs_list:
            # 如果有子目录
            subdir_modpath = os.path.join(root_dir, i, "mod")
            print(f"subdirectory mod path: {subdir_modpath}")
            ##检测是否下面有tokenizer 目录， 模型三件套：
            model_so = subdir_modpath + ".so"
            model_params = subdir_modpath + ".params"
            model_json = subdir_modpath + ".json"
            assert os.path.isfile(model_so), f"{i}模型so 不存在，{model_so}"
            assert os.path.isfile(model_params), f"{i}模型params 不存在，{model_params}"
            assert os.path.isfile(model_json), f"{i}模型json 不存在，{model_json}"
            model_tokenizer = subdir_modpath[:-3] + "tokenizer"
            assert os.path.exists(model_tokenizer), f"{i}tokenizer 目录不存在"
        return True
    src_scheme, src_root = parse_uri(model_uri)

    if src_scheme == "file":
        if not os.path.isabs(src_root):
            raise ValueError(f"Model URI cannot be a relative path: {model_uri}")
        return os.path.exists(src_root)
    else:
        # TODO: handle other schemes.
        return True