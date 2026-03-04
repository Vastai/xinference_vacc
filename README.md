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

https://github.com/xorbitsai/xoscar/pull/177

https://github.com/xorbitsai/xoscar/pull/174


## 3. 测试平台

- 以下为本指南测试使用的平台信息, 供参考
    ```
    os: Ubuntu-22.04.3-LTS-x86_64
    cpu: Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz
    gpu: VA16 / VA1L / VA10L
    torch: 2.8.0+cpu
    torch-vacc: 1.3.4.1081
    vllm: 0.11.0+.cpu
    vllm-vacc: 0.11.0.1081
    driver: 00.26.01.12 d3_3_v2_9_a3_1 eda8215 20260112
    docker: 28.1.1
    ```

## 4. 环境准备

> [!TIP]
> - 章节`4.1/4.2/4.3`, 用户可根据情况选择其中一种选择。

### 4.1 基于基础镜像制作Xinference
> [!NOTE]
> - `vllm_vacc`基础镜像内已包含`torch/vllm`等相关依赖
> - `xinference`目前最新版本是`2.1.0`
> - 需指定适当的`--shm-size`虚拟内存

1. 根据不同架构获取vllm_vacc基础镜像
    ```bash
    sudo docker pull harbor.vastaitech.com/ai_deliver/vllm_vacc:VVI-26.02
    sudo docker pull harbor.vastaitech.com/ai_deliver/vllm_vacc:VVI-26.02_arm
    ```

2. 启动容器
    ```bash
    sudo docker run -it \
        --privileged=true \
        --shm-size=256g \
        --name vllm_service \
        --ipc=host \
        --network=host \
        harbor.vastaitech.com/ai_deliver/vllm_vacc:VVI-26.02 bash
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
  sudo docker build -t xinference_vacc:VVI-26.02 .
  ```
如果使用arm架构，需要先修改Dockerfile 的基础镜像。
  ```bash
FROM harbor.vastaitech.com/ai_deliver/vllm_vacc:VVI-26.02_arm
  ```

### 4.3 拉取完整镜像

根据不同的架构，拉取完整镜像

  ```bash
  sudo docker pull harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-26.02
  sudo docker pull harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-26.02_arm
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
      --shm-size=256g \
      --name xinference_service \
      --ipc=host \
      --network=host \
      --entrypoint bash \
      harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-26.02 
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

5. 浏览器输入 `http://${xinference_host}:port`即可部署模型。详细可参考[Xorbits Inference 手册](https://github.com/xorbitsai/inference/blob/main/README_zh_CN.md)。


## 部署说明
等待xinference-local 启动好后, 我们用webui 方式部署。  

举例说明 Embedding bge-m3, 部署方式用tp1, 单副本, 部署在 die 0 上面。注意填写好模型在容器的目录。

![Alt text](./images/index/image-1.png)
![Alt text](./images/index/image-2.png)
这里注意要传递tensor_parallel_size 1, 和启动模型最大长度8192
![Alt text](./images/index/image-3.png)
然后可以查看状态
![Alt text](./images/index/image-4.png)

1. 多副本

如果要部署多个replica, 选择的tp 模式要和gpu index对齐。  

假如bge-m3 要部署2个副本, tp 2 的方式, 那么gpu index 需要写四个, 比如4,5,6,7。  

规则如下：  
  gpu index: GPU ID列表。列表数= TP * instance_nums。  

  如果是TP=2, instance_nums=2, 列表数= 2 * instance_nums, 可设置为 0,1,2,3。  

  如果是TP=4,  instance_nums=2, 列表数= 2 * instance_nums, 可设置为 0,1,2,3,4,5,6,7。  

  如果是TP=16, instance_nums=1, 列表数= 1 * instance_nums, 可设置为 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15。  

对于有些情况下, 比如您想从gpu index 0开始启动, 可以直接写GPU_counter per worker 然后配上副本数, 也可以生效。  
![Alt text](./images/index/image.png)  
这样的话, 您就不用手敲了。  

相当于, gpu indexs 为  

0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,  

33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63。  

但是假如您想要在特定某些gpu index加载模型, 那就要指定填写了, 并且需要保证gpu index 的连续性。  

2. hybrid模型：  

假如我们要启动一个Deepseek-V3.1 模型。  
  他是hybrid【可以选择开启或者不开思考模式】。  
  开启思考的话, 也可以选择是否要开启parse reasoning content【从输出中提取思考内容】。
  ![Alt text](./images/index/image-7.png)

启动vllm config 如下:  
  tensor_parallel_size: 张量并行数 
  enforce_eager: true  
  max_model_len: 模型最大上下文  
  max_num_seqs: 单例模型支持的最大并发数。vllm_vacc 对于LLM/VLM 模型是4。 

3. 特殊配置：

对于DS 系列, 如果要开启MTP, 就需要填充字段speculative_config。  
  speculative_config:{'method': 'deepseek_mtp', 'num_speculative_tokens': 1}
  ![Alt text](./images/index/image-5.png)

对于Qwen3 hybrid 系列： 如果模型原生支持扩充上下文。可以填充字段rope_scaling。 
  rope_scaling:{'rope_type': 'yarn', 'factor': 4.0, 'original_max_position_embeddings': 32768}
  ![Alt text](./images/index/image-6.png)


## 部署集群 
条件：
假设我们有两台机器，分别是10.24.73.25/10.24.73.23, 每台机器都满足条件, 镜像，模型已经准备好，16张VA16。
一台作为主集群入口（supervisor进程的 webui 入口）, 另一台是worker 进程加载模型。

我们想要加载4个Deepseek-V3.1-terminus 模型服务，并通过一个supervisor入口来调度请求。 
这里，我们选择在10.24.73.25 启动 supervisor + 2worker 进程，并选择9997端口作为supervisor 入口。
我们要在10.24.73.25执行启动容器命令。这时，这台机器并没有加载模型。只是启动了supervisor + 2worker 进程。

1. 10.24.73.25 启动容器并映射模型所在路径。
  ```bash
  sudo docker run -it \
      --privileged=true \
      -v /models:/models \
      --shm-size=256g \
      --name xinference_supervisor \
      --ipc=host \
      --network=host \
      --entrypoint bash \
      harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-26.02 
  ```
2. 使用screen工具查看日志。

 ```bash
  screen -S xinference_supervisor
 ```

3. 启动xinference-supervisor，假设端口为9997, 以及两个worker 进程。可以写一个bash 脚本把这些命令打包一起。
 ```bash
    echo "=================== start run supervisor ========================="
    xinference-supervisor -H 10.24.73.25 -p 9997 & 2>&1 | tee xinference.log &
    until curl -s http://10.24.73.25:9997/status; do
      sleep 5
    done

    echo "==================== STARTING WORKERS ========================="
    
    echo "==================== STARTING WORKERS 1 ========================="
    VACC_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31'  xinference-worker -H 10.24.73.25 -e http://10.24.73.25:9997 &
    sleep 10

    echo "==================== STARTING WORKERS 2========================="
    VACC_VISIBLE_DEVICES='32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63'  xinference-worker -H 10.24.73.25 -e http://10.24.73.25:9997 &
    sleep 10
 ```

4. 另一台 10.24.73.23 启动容器并映射模型所在路径。(保证两个容器内，模型所在的映射路径是相同的，比如都是/models/Deepseek，否则启动会失败)
  ```bash
  sudo docker run -it \
      --privileged=true \
      -v /models:/models \
      --shm-size=256g \
      --name xinference_worker \
      --ipc=host \
      --network=host \
      --entrypoint bash \
      harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-26.02 
  ```
5. 使用screen工具查看日志。

 ```bash
  screen -S xinference_worker
 ```
6. 在另一台机器 10.24.73.23 启动两个worker 进程，并绑定到这个supervisor。 可以通过bash 脚本来操作。
 ```bash
    echo "==================== STARTING WORKERS 3 ========================="
    VACC_VISIBLE_DEVICES='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31'  xinference-worker -H 10.24.73.23 -e http://10.24.73.25:9997 &
    sleep 10

    echo "==================== STARTING WORKERS 4========================="
    VACC_VISIBLE_DEVICES='32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63'  xinference-worker -H 10.24.73.23 -e http://10.24.73.25:9997 &
    sleep 10
 ```
Note: 可通过环境变量 `VACC_VISIBLE_DEVICES` 指定容器内可见的Die 列表，其功能与 NVIDIA 环境中的 CUDA_VISIBLE_DEVICES 相同。例如，启动容器时使用 -e VACC_VISIBLE_DEVICES=0,1,2,3。 即可使容器仅识别并使用前四个Die。为保障 vLLM 框架在多进程数据加载与通信时的稳定性，可通过 --shm-size 参数为容器分配充足的共享内存（Shared Memory）。

7. 浏览器输入 `http://10.24.73.25:9997`即可通过webui部署模型。我们有四个worker, 对应四个副本。填好模型路径和相关配置，耐心等待，模型就会在两台机器上启动了。每个worker 会加载自己的模型实例。
详细可参考[Xorbits Inference 手册](https://github.com/xorbitsai/inference/blob/main/README_zh_CN.md)。

强烈推荐用Webui可视化部署模型, 运行服务稳定, 精度与NVIDIA GPU基本一致。

此外，Xinference中通过`enable_xavier=True`启用的VLLM多副本共享KV缓存功能，目前仅支持英伟达硬件平台。
详细可以参考。
[Xinference 关于Xavier说明](https://inference.readthedocs.io/zh-cn/latest/getting_started/using_xinference.html#run-xinference-locally)。


## 声明
- `Xinference`采用[Apache License 2.0](https://inference.readthedocs.io/zh-cn/latest/getting_started/installation.html)。
- `Xinference_vacc`遵循[Apache 2.0](LICENSE)许可证许可。
- Additional components and integrations: Copyright © 2024-2025 vastaitech.

