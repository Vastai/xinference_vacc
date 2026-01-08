## 0. 瀚博半导体

![vastaitech](https://github.com/Vastai/VastModelZOO/blob/main/images/index/logo.png?raw=true)

- 官方网址：https://www.vastaitech.com
- 模型中心：https://github.com/Vastai/VastModelZOO

## 1. 官方支持
Xinference（Xorbits Inference）是一个性能强大且功能全面的开源分布式推理框架，旨在简化本地和云端模型的部署与管理，支持多样化的模型类型（如大语言模型、嵌入模型和多模态模型）和硬件加速（支持CPU、GPU等）。Xinference不仅可以在本地运行推理模型，还支持在分布式集群环境下高效推理，可以轻松扩展以处理更大规模的推理任务。

xinference 目前适配了瀚博硬件，支持使用瀚博硬件设备进行LLM系列、Embedding系列、Rerank系列, VL系列模型的部署和推理。 

- ✨依赖xinference官方仓库，零代码修改，可实现用xinference 平台启动模型在VACC硬件下推理
- https://github.com/xorbitsai/inference
 
集成具体PR 如下：  

https://github.com/xorbitsai/inference/pull/4382  

https://github.com/xorbitsai/inference/pull/4385  

改社区bug 如下：
https://github.com/xorbitsai/inference/pull/4422  

https://github.com/xorbitsai/inference/pull/4370  

https://github.com/xorbitsai/inference/pull/4332

https://github.com/xorbitsai/xoscar/pull/177

https://github.com/xorbitsai/xoscar/pull/174


## 2. 测试平台

- 以下为本指南测试使用的平台信息，供参考
    ```
    os: Ubuntu-22.04.3-LTS-x86_64
    cpu: Intel(R) Xeon(R) Platinum 8358 CPU @ 2.60GHz
    gpu: VA16 / VA1L / VA10L
    torch: 2.8.0+cpu
    torch-vacc: 1.3.3.777
    vllm: 0.11.1.dev0+gb8b302cde.d20251030.cpu
    vllm-vacc: 0.11.0.777
    driver: 00.25.12.30 d3_3_v2_9_a3_1 a76bf37 20251230
    docker: 28.1.1
    ```

## 3. 环境准备

> [!TIP]
> - 步骤`3.1/3.2/3.3`，可任选其一使用

### 3.1 从基础镜像安装

- 获取vllm_vacc基础镜像
    ```bash
    sudo docker pull harbor.vastaitech.com/ai_deliver/vllm_vacc:VVI-25.12.SP2
    ```

- 启动容器
    ```bash
    sudo docker run -it \
        --privileged=true \
        --shm-size=256g \
        --name vllm_service \
        --ipc=host \
        --network=host \
        harbor.vastaitech.com/ai_deliver/vllm_vacc:VVI-25.12.SP2 bash
    ```

- 安装Xinference

   - 参考官方文档安装：[README_zh-CN.md#安装-xinference](https://inference.readthedocs.io/zh-cn/latest/getting_started/installation.html)

        ```bash
        # 启动容器
        # sudo docker exec -it vllm_service bash
        
        # 可选pypi源
        # https://mirrors.163.com/pypi/simple/
        # https://mirrors.aliyun.com/pypi/simple/
        # https://pypi.mirrors.ustc.edu.cn/simple/
        # https://pypi.tuna.tsinghua.edu.cn/simple/
        # https://mirror.baidu.com/pypi/simple

        # 通过源码安装
        pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
        git clone https://github.com/xorbitsai/inference.git
        cd xinference
        git checkout 00957020f29ee8ffd918eec877833d7904966ff0
        pip install -e .[vllm]

        # 或使用pip安装Xinference, 但是由于最新tag 1.16.0 并没有完全包括PR相关的改动, 因此需要手动在合入修改，暂不推荐。
        # 等发新tag，可以选择此操作。
        pip install -U "xinference==1.16.0" -i https://mirrors.aliyun.com/pypi/simple
        https://github.com/xorbitsai/inference/commit/00957020f29ee8ffd918eec877833d7904966ff0
        
        ```

### 3.2 编译完整镜像

- 编译镜像, 根据平台来选择

  > - [Dockerfile](./dockerfile/Dockerfile)
  
  ```bash
  cd dockerfile
  sudo docker build -t xinference_vacc:VVI-25.12.SP2 .
  ```

- 启动容器
    ```bash
    sudo docker run -it \
        --privileged=true \
        --shm-size=256g \
        --name xinference_service \
        --ipc=host \
        --network=host \
        xinference_vacc:VVI-25.12.SP2 bash
    ```


### 3.3 拉取完整镜像

- 获取完整镜像,根据平台选择

  ```bash
  sudo pull harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-25.12.SP2
  sudo pull harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-25.12.SP2_arm
  ```

- 启动容器
  ```bash
  sudo docker run -it \
      --privileged=true \
      --shm-size=256g \
      --name xinference_service \
      --ipc=host \
      --network=host \
      harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-25.12.SP2 bash
  ```

> [!NOTE]
> - `vllm_vacc`基础镜像内已包含`torch/vllm`等相关依赖
> - 截至`2025/12/31`，`VastAI`已支持`xinference`至最新版本`1.16.0`, suppport vLLM engine
> - 和`NVIDIA`硬件下`CUDA_VISIBLE_DEVICES`类似；在`VastAI`硬件中可以使用`VACC_VISIBLE_DEVICES`指定`可见计算卡ID`，如`-e VACC_VISIBLE_DEVICES=0,1,2,3`
> - 需指定适当的`--shm-size`虚拟内存


## 4. 目前支持情况以及相关部署脚本样例
## LLM Models
- DeepSeek-V3
- DeepSeek-V3-0324
- DeepSeek-V3.1
- DeepSeek-V3.1-Terminus
- DeepSeek-R1
- DeepSeek-R1-0528
- Qwen3-30B-A3B-FP8
- Qwen3-30B-A3B-Instruct-2507-FP8
- Qwen3-30B-A3B-Thinking-2507-FP8
- Qwen3-235B-A22B-Instruct-2507
- Qwen3-235B-A22B-Thinking-2507
## Embedding Models
- bge-m3
- Qwen3-0.6B-Embedding
## Rerank Models
- bge-reranker-v2-m3
- Qwen3-0.6B-Rerank
## VL Models
- Qwen3-VL-30B-A3B-Instruct-FP8
- Qwen3-VL-30B-A3B-Thinking-FP8
## 附录表格
| 模型 | tensor parallel size(启动单实例) |
|------------|-------|
| DS Familiy | tp32 | 
| Qwen3 235B | tp16 | 
| Qwen3 30B  | tp2, tp4 | 
| text2vec  |  tp1, tp2, tp4 |  
| Qwen3-VL  | tp2, tp4 | 
## 备注
针对DS 系列，可以开启MTP， 对于非MTP 的启动，可以支持最大输入100K。其中需要pipeline data size 2。  

针对Qwen3 235B系列，可以支持最大输入100K。
## 用webui 部署
我们在物理机上面把模型准备好，映射到容器里面 
举例说明，假如要用9997端口去启动xinference-local。 
- 启动容器
  ```bash
  sudo docker run -it \
      --privileged=true \
      -v /models:/models \
      --shm-size=256g \
      --name xinference_service \
      --ipc=host \
      --network=host \
      harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-25.12.SP2 bash
  xinference-local -H 0.0.0.0 -p 9997 & 
  ```
- 浏览器输入 `http://${supervisor_host}:port`
- 通过 `Cluster Information` 页面查看集群信息
- 通过 `Running Models` 页面查看启动的模型
- `curl 'http://localhost:port/v1/models'`
- 更加丰富的介绍可以看社区说明。  

https://github.com/xorbitsai/inference/blob/main/README_zh_CN.md
https://inference.readthedocs.io/zh-cn/latest/getting_started/using_xinference.html#run-xinference-locally

这里，我们举例部署Embedding bge-m3，部署方式用tp1, 单副本，部署在 die 0 上面。注意填写好模型在容器的目录。
![Alt text](./images/index/image-1.png)
![Alt text](./images/index/image-2.png)
这里注意要传递tensor_parallel_size 1，和模型最大长度8192
![Alt text](./images/index/image-3.png)
然后可以查看状态
![Alt text](./images/index/image-4.png)

这里注意多副本的概念。如果要部署多个replica, 那么对应的gpu index 要对齐。
假如bge-m3 要部署2个副本，tp 2 的方式，那么gpu index 需要写四个，比如4,5,6,7
gpu index: GPU ID列表。列表数= TP * instance_nums。

例如，TP=2，instance_nums=2，列表数= 2 * instance_nums，可设置为 0,1,2,3。  

如果是TP=4， instance_nums=2，列表数= 2 * instance_nums，可设置为 0,1,2,3,4,5,6,7。  

如果是TP=16，instance_nums=1, 列表数= 1 * instance_nums，可设置为 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15。

以下是对应的模型名字
| 模型名字 | 模型目录| 
|-------|-------|
| deepseek-v3 | DeepSeek-V3、DeepSeek-V3-0324 | 
| deepseek-r1 | DeepSeek-R1、DeepSeek-R1-0528 |
| DeepSeek-V3.1 |DeepSeek-V3.1-Terminus、DeepSeek-V3.1 |
| qwen3 | Qwen3-30B-A3B-FP8、Qwen3-30B-A3B-GPTQ-Int4、 Qwen3-235B-A22B-FP8|
| Qwen3-Instruct | Qwen3-30B-A3B-Instruct-2507-FP8、Qwen3-235B-A22B-Instruct-2507 |
| Qwen3-Thinking | Qwen3-30B-A3B-Thinking-2507-FP8, Qwen3-235B-A22B-Thinking-2507 |
| Qwen3-VL-Instruct | Qwen3-VL-30B-A3B-Instruct-FP8 |
| Qwen3-VL-Thinking | Qwen3-VL-30B-A3B-Thinking-FP8 |
| Qwen3-Embedding-0.6B| Qwen3-Embedding-0.6B|
| Qwen3-Reranker-0.6B | Qwen3-Reranker-0.6B |
| bge-m3 | bge-m3 |
| bge-reranker-v2-m3 | bge-reranker-v2-m3 |

### function call 测试
非流式：    
function_call\nonstream_tool_calls.py
```{code-block}
python3 nonstream_tool_calls.py \
--host 127.0.0.1 \
--port 9994 \
--model-name Qwen3-Instruct
```
流式：  
function_call\stream_tool_calls.py
```{code-block}
python3 stream_tool_calls.py \
--host 127.0.0.1 \
--port 9994 \
--model-name Qwen3-Instruct
```
其中根据需要修改模型名字，端口号。  


### 模型最大上下文长度限制：

针对 DeepSeek-V3/R1/V3.1 系列模型，模型最大上下文长度为 64K。  

针对 Qwen3 系列模型，如果 TP 为 2，模型最大上下文长度为 64K；如果TP 为 4或16，模型最大上下文长度为 128K。  

针对 Embedding/Rerank模型， bge-m3, bge-reranker-v2-m3 默认启动最大长度8192。  

Qwen3-Embedding-0.6B 最大长度 65536, Qwen3-Rerank-0.6B 默认最大长度 40960
> Note:
单模型同时支持最大并发数为 4。如果有多并发需求，可以用多副本
对于超出上下文长度的请求，服务端会拦截不做处理，客户端需自行校验请求长度。  

对于text2vec 模型，尽管xinference 有内部auto batch 的聚合功能，但在低并发情况下，性能是要稍低于用Vllm serve 原生方式。  

原因是Vllm 社区，对于text2vec 模型，vllm asyncEngine 对外没有暴露类似于generate的接口。  
只能用同步的LLM 方式启动的。这个和显卡无关，这个在CPU上cores 利用率也有一点差异。
