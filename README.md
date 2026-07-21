## 1. 瀚博半导体

![vastaitech](https://github.com/Vastai/VastModelZOO/blob/main/images/index/logo.png?raw=true)

- 官方网址：https://www.vastaitech.com
- 模型中心：https://github.com/Vastai/VastModelZOO

## 2. 官方支持

Xinference（Xorbits Inference）【Copyright © 2022-2023 XProbe Inc.】是一个性能强大且功能全面的开源分布式推理框架, 旨在简化本地和云端模型的部署与管理, 支持多样化的模型类型（如大语言模型、嵌入模型和多模态模型）和硬件加速（支持CPU、GPU等）。Xinference不仅可以在本地运行推理模型, 还支持在分布式集群环境下高效推理, 可以轻松扩展以处理更大规模的推理任务。


xinference 目前适配了瀚博硬件, 支持使用瀚博硬件设备进行LLM系列、Embedding系列、Rerank系列, VLM系列模型的部署和推理。
Xinference_vacc 项目是针对这个集成，帮助用户更好的使用瀚博半导体的产品，并且更加完备的介绍瀚博对于Xinference社区的共享。

✨基于Xinference框架，用户无需代码修改即可将模型部署至VACC硬件进行推理。VastAI仅支持vLLM engine启动。 

Xinference仓库地址：
https://github.com/xorbitsai/inference

集成具体PR 如下：  

https://github.com/xorbitsai/inference/pull/4382  

https://github.com/xorbitsai/inference/pull/4385  

改社区bug 如下：  

https://github.com/xorbitsai/inference/pull/4422  

https://github.com/xorbitsai/inference/pull/4370  

https://github.com/xorbitsai/inference/pull/4332

https://github.com/xorbitsai/inference/pull/4454

https://github.com/xorbitsai/inference/pull/4486

https://github.com/xorbitsai/inference/pull/4569

https://github.com/xorbitsai/inference/pull/4684

https://github.com/xorbitsai/inference/pull/4683

https://github.com/xorbitsai/inference/pull/4678

https://github.com/xorbitsai/inference/pull/4747

https://github.com/xorbitsai/inference/pull/4752

https://github.com/xorbitsai/xoscar/pull/177

https://github.com/xorbitsai/xoscar/pull/174


## 3. 依赖软件

- 基于`瀚博半导体VastAI`的硬件产品使用`xinference_vacc`前，需联系销售代表获取`瀚博开发者中心`版本权限

- 访问[瀚博开发者中心](https://developer.vastaitech.com/downloads/vvi?version_uid=)，获取`VVI(Vastai Versatilve Inference)`部署软件包


## 4. 环境准备
 VVI-version表示生产环境的版本号
> [!TIP]
> - 章节`4.1/4.2/4.3`, 用户可根据情况选择其中一种选择。

### 4.1 基于基础镜像制作Xinference
> [!NOTE]
> - `vllm_vacc`基础镜像内已包含`torch/vllm`等相关依赖
> - 需指定适当的`--shm-size`虚拟内存

1. 根据不同架构获取vllm_vacc基础镜像
    ```bash
    sudo docker pull harbor.vastaitech.com/ai_deliver/vllm_vacc:<VVI-Version>
    sudo docker pull harbor.vastaitech.com/ai_deliver/vllm_vacc:<VVI-Version>_arm
    ```

2. 启动容器
    ```bash
    sudo docker run -it \
        --privileged=true \
        --shm-size=512g \
        --name vllm_service \
        --ipc=host \
        --network=host \
        harbor.vastaitech.com/ai_deliver/vllm_vacc:<VVI-Version> bash
    ```

3. 安装Xinference

   - 参考官方文档安装：[README_zh-CN.md#安装-xinference](https://inference.readthedocs.io/zh-cn/latest/getting_started/installation.html)

        ```bash
        # 启动容器
        sudo docker exec -it vllm_service bash
        # 引入环境变量
        export XINFERENCE_SSE_PING_ATTEMPTS_SECONDS=864000
        export VLLM_ENGINE_ITERATION_TIMEOUT_S=864000
        export XOSCAR_CPU_AFFINITY=1
        export XINFERENCE_ENABLE_VIRTUAL_ENV=0
        export XINFERENCE_RERANK_EMPTY_CACHE_COUNT=200
        export XINFERENCE_EMBEDDING_EMPTY_CACHE_COUNT=200
        export XINFERENCE_EMBEDDING_EMPTY_CACHE_TOKENS=81920  
        
        # 通过轮子包安装
        pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
        pip install xinference
        pip install qwen_omni_utils
        pip install qwen-vl-utils
        pip install mineru_vl_utils
        pip install -U "mineru[core]==2.7.0"
        ```

### 4.2 基于Dockerfile制作Xinference

编译镜像, 默认使用x86架构。
  > - [Dockerfile](./dockerfile/Dockerfile)

  ```bash
  cd dockerfile
  sudo docker build -t xinference_vacc:<VVI-Version> .
  ```
如果使用arm架构，需要先修改Dockerfile 的基础镜像。
  ```bash
FROM harbor.vastaitech.com/ai_deliver/vllm_vacc:<VVI-Version>_arm
  ```

### 4.3 拉取完整镜像

根据不同的架构，拉取完整镜像

  ```bash
  sudo docker pull harbor.vastaitech.com/ai_deliver/xinference_vacc:<VVI-Version>
  sudo docker pull harbor.vastaitech.com/ai_deliver/xinference_vacc:<VVI-Version>_arm
  ```

## 5. 模型列表

关于支持的模型以及每个模型的具体配置信息，如最大上下文长度, 输入限制和支持的张量并行度, 请查看[模型使用限制](https://github.com/Vastai/VastModelZOO/blob/develop/docs/vllm/usage_limits.md)。


## 部署模型 

1. 准备待部署的模型。
2. 启动容器并映射模型所在路径。

  ```bash
  sudo docker run -it \
      --privileged=true \
      -v /models:/models \
      --shm-size=512g \
      --name xinference_service \
      --ipc=host \
      --network=host \
      --entrypoint bash \
      harbor.vastaitech.com/ai_deliver/xinference_vacc:<VVI-Version> 
  ```
3. 使用screen工具查看日志。

 ```bash
  screen -S xinference
 ```

4. 在本地启动xinference-local，假设端口为9997。
 ```bash
   xinference-local -H 0.0.0.0 -p 9997 2>&1 | tee xinference.log & 
 ```
Xinference启动方式详细可参考[Xinference 入门指南](https://inference.readthedocs.io/zh-cn/latest/getting_started/using_xinference.html#run-xinference-locally)。

Note: 可通过环境变量 `VACC_VISIBLE_DEVICES` 指定容器内可见的Die 列表，其功能与 NVIDIA 环境中的 CUDA_VISIBLE_DEVICES 相同。例如，启动容器时使用 -e VACC_VISIBLE_DEVICES=0,1,2,3。 即可使容器仅识别并使用前四个Die。为保障 vLLM 框架在多进程数据加载与通信时的稳定性，可通过 --shm-size 参数为容器分配充足的共享内存（Shared Memory）。
如果是 qwen2 audio 7b 或者qwen2 audio 7b instruct, 需要VNNL_CONV1D_DLC=1配置上。
详细可参考
https://github.com/Vastai/VastModelZOO/tree/main/alm/qwen2_audio/vllm

5. 浏览器输入 `http://${xinference_host}:port`即可部署模型。详细可参考[Xorbits Inference 手册](https://github.com/xorbitsai/inference/blob/main/README_zh_CN.md)  。


强烈推荐用Webui可视化部署模型, 运行服务稳定, 精度与NVIDIA GPU基本一致。

此外，Xinference中通过`enable_xavier=True`启用的VLLM多副本共享KV缓存功能，目前仅支持英伟达硬件平台。
详细可以参考。
测试qwen3-reranker 系列模型，xinference社区默认会加模版，可以通过XINFERENCE_QWEN3_RERANK_TEMPLATE变量来控制关
[Xinference 关于Xavier说明](https://inference.readthedocs.io/zh-cn/latest/getting_started/using_xinference.html#run-xinference-locally)。


## 声明
- `Xinference`采用[Apache License 2.0](https://inference.readthedocs.io/zh-cn/latest/getting_started/installation.html)。
- `Xinference_vacc`遵循[Apache 2.0](LICENSE)许可证许可。
- Additional components and integrations: Copyright © 2024-2025 vastaitech.
