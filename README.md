<div id=top align="center">

![logo](./images/index/logo.png)
[![License](https://img.shields.io/badge/license-Apache_2.0-yellow)](LICENSE)
[![company](https://img.shields.io/badge/vastaitech.com-blue)](https://www.vastaitech.com/)


</div>

---
# xinference on vacc
# 简介


Xinference（Xorbits Inference）是一个性能强大且功能全面的开源分布式推理框架，旨在简化本地和云端模型的部署与管理，支持多样化的模型类型（如大语言模型、嵌入模型和多模态模型）和硬件加速（支持CPU、GPU等）。Xinference不仅可以在本地运行推理模型，还支持在分布式集群环境下高效推理，可以轻松扩展以处理更大规模的推理任务。


xinference_vacc 是适配了瀚博硬件设备的分布式推理框架，支持使用瀚博硬件设备进行LLM系列、Embedding系列、Rerank系列模型的部署和推理。

## Engine
- [`20250826`]: suppport vLLM engine, vsx

## Models
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
- Embedding (supported by Vastai ModelZoo)
- Rerank (supported by Vastai ModelZoo)

## Quick Start
```bash
# whl
python setup.py bdist_wheel
# source
pip install -v .
```
## 准备镜像（确保有外网，直接下载公开镜像） 
x86平台：
docker pull harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-25.11
arm平台：
docker pull harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-25.11_arm
## 准备模型
根据您的需要，准备好模型。下载到服务器。如果是需要Vastai ModelZoo支持的Emb/rerank 模型，请找瀚博相关技术人员获取。
```
example目录结构如下所示
example
├── ds3
|   ├── cluster
│   ├── MTP
│   └── nonMTP
├── emb-rerank
├── qwen3
│   ├── Qwen3-30B-A3B-FP8
│   ├── Qwen3-30B-A3B-Instruct-2507-FP8
│   └── Qwen3-30B-A3B-Thinking-2507-FP8
│	├── Qwen3-235B-A22B-Instruct-2507
│   └── Qwen3-235B-A22B-Thinking-2507
│
└── install_docker_compose.sh
```

目录/文件	说明
| 目录 | 意义| 
|-------|-------|
| ds3   | DeepSeek-V3 或 DeepSeek-R1 系列模型服务的 Docker Compose 文件及测试脚本。|
| ds3/cluster | （可选）用于跨机器启动 DeepSeek 模型的配置。 | 
| ds3/MTP | 包含引入多令牌预测技术 (Multi-Token Prediction) 的配置，用于提升模型推理性能。 |
| ds3/nonMTP | 未引入多令牌预测技术 (MTP) 的标准部署配置。|
| emb-rerank | 启动 Embedding 或 Rerank 系列模型服务的 Docker Compose 文件及测试脚本。|
| qwen3      | 启动 Qwen3 系列模型服务的 Docker Compose 文件及测试脚本。|

| 文件 | 意义 |
|-------|-------|
| xxx_1model.yaml | 部署 1个 模型实例的 Docker Compose 配置文件。|
| xxx.yaml        | 部署 2个 模型实例的 Docker Compose 配置文件。|
| tp2.yaml | 部署 Qwen3-30B模型实例的 Docker Compose 配置文件, TP2。 |
| tp4.yaml        | 部署 Qwen3-30B模型实例的 Docker Compose 配置文件, TP4。 |
| tp16.yaml | 部署 Qwen3-235B模型实例的 Docker Compose 配置文件, TP16。 |

**安装 Docker Compose。** 

如果已安装，可跳过该步骤。

```shell
./install_docker_compose.sh
```

**离线安装 Docker。** 

如果已安装，可跳过该步骤。

```shell
# 解压
tar -xzvf docker-28.0.6.tgz

# 复制二进制文件到系统路径
sudo cp docker/* /usr/bin/

# 创建 systemd 服务文件
sudo vi /etc/systemd/system/docker.service
[Unit]
Description=Docker Application Container Engine
Documentation=https://docs.docker.com
After=network-online.target firewalld.service
Wants=network-online.target

[Service]
Type=notify
ExecStart=/usr/bin/dockerd
ExecReload=/bin/kill -s HUP $MAINPID
LimitNOFILE=infinity
LimitNPROC=infinity
LimitCORE=infinity
TasksMax=infinity
TimeoutStartSec=0
Delegate=yes
KillMode=process
Restart=on-failure
StartLimitBurst=3
StartLimitInterval=60s

[Install]
WantedBy=multi-user.target
然后启动服务
sudo systemctl daemon-reload
sudo systemctl start docker
sudo systemctl enable docker
```

# 使用说明

本章节主要描述如何通过 xinference_vacc 启动模型服务。

## 启动 DeepSeek 系列模型服务

通过 xinference_vacc 启动 DeepSeek-V3 或 DeepSeek-R1系列模型，其步骤如下所示。

**前提条件**

example/ds3 的每个子目录下，都有.env 变量
用于配置yaml 中的变量。
```shell
# 模型目录的路径
HOST_DATA_DIR=/FS03/wyl_data/workspace/weights
# 镜像设置
IMAGE=harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-25.11
## 如果是arm 平台，公版的镜像是 harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-25.11_arm
# 参数设置
model_name=deepseek-v3
model_directory=DeepSeek-V3.1
```
其中，HOST_DATA_DIR表示存放模型目录的路径。
具体模型目录是model_directory来指定。
IMAGE 表示使用的镜像名称。
这里要注意的是模型名字。
| 模型名字(不可更改） | 模型目录| 
|-------|-------|
| deepseek-v3 | DeepSeek-V3.1-Terminus、DeepSeek-V3.1、DeepSeek-V3、DeepSeek-V3-0324 | 
| deepseek-r1 | DeepSeek-R1、DeepSeek-R1-0528 |


**步骤 1.** 
根据实际情况选择“example/ds3/MTP(或者nonMTP)/xxx.yaml”文件, 并修改.env

其中，“xxx”为模型名，请根据实际情况替换。

**步骤 2.**  启动模型服务。

```shell
cd /home/username/example/ds3/MTP(或者nonMTP)
docker-compose -f xxx.yaml up -d 
```

**步骤 3.** 检查模型服务是否启动成功。

- 如果模型为 DeepSeek-V3系列模型，按如下步骤执行。


1. 修改“example/ds3/v3chat.py”中“base_url”。

> “base_url”为 模型服务地址，格式为[http://IP:Port/v1](http://IP:Port/v1)。其中，IP为 模型服务IP地址，请根据实际情况设置。“Port”为模型服务端口,可在“example/ds3/xxx.yaml”中查看“ports”参数的值确认其端口号。



```{code-block}
from openai import OpenAI
client = OpenAI(base_url="http://localhost:9997/v1", api_key="EMPTY")

response = client.chat.completions.create(
  model="deepseek-v3",
  messages=[{"role": "user", "content": "中国直辖市是哪里"}],
  temperature=0.5,
)
print(response.choices[0].message.content)
```

2. 执行测试脚本。
```shell
cd /home/username/example/ds3
python3 v3chat.py
```



- 如果模型为 DeepSeek-R1系列模型，则执行如下步骤执行。

1. 修改“example/ds3/r1chat.py”中“base_url”。

> “base_url”为 模型服务地址，格式为[http://IP:Port/v1](http://IP:Port/v1)。其中，IP为 模型服务IP地址，请根据实际情况设置。“Port”为模型服务端口，可在“example/ds3/xxx.yaml”中查看“ports”参数的值确认其端口号。



```{code-block}
from openai import OpenAI
client = OpenAI(base_url="http://localhost:9997/v1", api_key="EMPTY")

response = client.chat.completions.create(
  model="deepseek-r1",
  messages=[{"role": "user", "content": "中国直辖市是哪里"}],
  temperature=0.5,
)
print(response.choices[0].message.content)
```

2. 执行测试脚本。

```shell
cd /home/username/example/ds3
python3 r1chat.py
```

## 跨机启动 DS3 系列模型服务
**（可选，根据您手上的资源来）**

	如果您手上有两台或者更多瀚博的一体机，并且网络能互通，而且您这边有多服务需求。
由于我们每台一体机最多只能部署两个DeepSeek 模型，您这边可以根据需要搭建集群。  
  

**前提条件**  

根据实际情况修改“example/ds3/cluster/*.yaml”文件中“volumes”参数，将其修改为实际模型权重文件夹所在路径。注意，多台机器的模型在物理机的绝对路径需要一致，才能跨机加载，这边建议可以用网盘。  

在cluster 目录下，是一个场景例子。这边做一下解释说明，可以根据您那边需要修改。  

场景：
	假设我们有两台机器，分别是10.24.73.25/10.24.73.23, 每台机器都满足条件  
	（镜像一致，模型已经准备好，16张VA16）  

我们想要加载4个Deepseek-V3.1 模型服务，并通过一个supervisor入口来调度请求. 
这里，我们选择在10.24.73.25 启动 supervisor + 2worker 进程，并选择9997端口作为supervisor 入口。
我们要在10.24.73.25执行启动容器命令。这时，这台机器并没有加载模型。只是启动了supervisor + 2worker 进程。
```shell
cd /home/username/example/ds3/cluster
docker-compose -f cluster.yaml up -d 
```
接着，我们在另一个机器10.24.73.23执行启动容器命令。  
这边启动2worker进程后，会执行加载模型和replica 4 副本的请求。
```shell
cd /home/username/example/ds3/cluster
docker-compose -f slave.yaml up -d 
```
等待一段时间后，4个模型启动好。
可以通过上面的步骤来测试服务了。这边的supervisor 是10.24.73.25，端口是9997.


## 启动 Qwen3 系列模型服务


通过 xinference_vacc 启动 Qwen3 系列模型，其步骤如下所示。  

**前提条件**

example/qwen3 的每个子目录下，都有.env 变量
用于配置yaml 中的变量。
```shell
# 模型目录的路径
HOST_DATA_DIR=/FS03/wyl_data/workspace/weights
# 镜像设置
IMAGE=harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-25.11
## 如果是arm 平台，公版的镜像是 harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-25.11_arm
# 参数设置
model_name=qwen3
model_directory=Qwen3-30B-A3B-FP8
GPU_PAIRS=16,17,18,19
instance_nums=2
```
其中，HOST_DATA_DIR表示存放模型目录的路径。具体模型目录是model_directory来指定。
IMAGE 表示使用的镜像名称。
- GPU_PAIRS: GPU ID列表。列表数= TP * instance_nums。  
例如，TP=2，instance_nums=2，列表数= 2 * instance_nums，可设置为 0,1,2,3 。  
如果是TP=4， instance_nums=2，列表数= 2 * instance_nums，可设置为 0,1,2,3,4,5,6,7。   
如果是TP=16，instance_nums=1, 列表数= 1 * instance_nums，可设置为 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15。
- instance_nums：实例数量。

| 模型名字（不可更改） | 模型目录|
|-------|-------|
| qwen3 | Qwen3-30B-A3B-FP8、Qwen3-30B-A3B-Instruct-2507-FP8、Qwen3-30B-A3B-Thinking-2507-FP8, Qwen3-235B-A22B-Instruct-2507, Qwen3-235B-A22B-Thinking-2507 |


**步骤 1.** 根据实际情况选择“example/qwen3/model_name/xxx.yaml”文件,
并修改.env 文件

其中，“model_name”为模型名称，“xxx”为tp2 或 tp4，请根据实际情况替换。

针对 Qwen3-30B 系列模型，当前TP仅支持 2 或 4 。如果是针对 Qwen3-235B，当前TP仅支持16

**步骤 2.**  启动模型服务。

```shell
cd /home/username/example/qwen3/model_name
docker-compose -f xxx.yaml up -d 
```

**步骤 3.** 检查模型服务是否启动成功。

1. 修改“example/qwen3/chat.py”中“base_url”。
> 其中，“base_url”为 模型服务地址，格式为[http://IP:Port/v1](http://IP:Port/v1)。其中，IP为 模型服务IP地址，请根据实际情况设置。“Port”为模型服务端口,可在“example/qwen3/xxx.yaml”中查看“ports”参数的值确认其端口号。

```{code-block}
from openai import OpenAI
client = OpenAI(base_url="http://localhost:9997/v1", api_key="EMPTY")

response = client.chat.completions.create(
  model="qwen3",
  messages=[{"role": "user", "content": "中国直辖市是哪里"}],
  temperature=0.5,
)
print(response.choices[0].message.content)
```

2. 执行测试脚本。
```shell
cd /home/username/example/qwen3
python3 chat.py
```

## 启动 Embedding 或 Rereank 系列模型服务

	通过 xinference_vacc 启动 Embedding 或 Rerank 系列模型，其步骤如下所示。  

**前提条件**  
	example/emb_rerank 的每个子目录下，都有.env 变量
用于配置yaml 中的变量。
```shell
# 模型目录的路径
EMB_DATA_DIR=/home/tonyguo/emb_models/bge-m3-vacc
RERANK_DATA_DIR=/home/tonyguo/emb_models/bge-reranker-v2-m3-vacc
# 镜像设置
IMAGE=harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-25.11
## 如果是arm 平台，公版的镜像是 harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-25.11_arm
# 参数设置
embedding_model_name=emb_vacc
embedding_GPUs=0,1
embedding_model_len=512
embedding_instance_nums=2
rerank_model_name=rerank_vacc
rerank_GPUs=2,3
rerank_instance_nums=2
```
其中，EMB_DATA_DIR,RERANK_DATA_DIR表示Vastai emb/rerank模型目录的路径。
模型可以找瀚博技术人员领取支持。  

举例说明，假如提供的【512,1024,2048,4096,8192】模型尺寸如下，
我们会把不同尺寸的模型都加载到每个指定的die 上， 然后根据用户请求长度，动态的去调度模型来处理。  

他们的目录结构是一样的，如下：
```shell
├── 512
│   ├── mod.json
│   ├── mod.params
│   ├── mod.so
│   └── tokenizer
├── 1024
│   ├── mod.json
│   ├── mod.params
│   ├── mod.so
│   ├── tokenizer
│   ├── vamc.env
│   └── vamc.yaml
├── 2048
│   ├── mod.json
│   ├── mod.params
│   ├── mod.so
│   ├── tokenizer
│   ├── vamc.env
│   └── vamc.yaml
├── 4096
│   ├── mod.json
│   ├── mod.params
│   ├── mod.so
│   ├── tokenizer
│   ├── vamc.env
│   └── vamc.yaml
├── 8192
│   ├── mod.json
│   ├── mod.params
│   ├── mod.so
│   ├── tokenizer
│   ├── vamc.env
│   └── vamc.yaml
```
其中，每个子目录的Tokenizer 文件夹，其文件结构如下所示。
```shell
├── tokenizer_config.json
├── tokenizer.json
└── vacc_config.json
```
vacc_config.json如下所示,用户可根据实际情况进行设置。

- batch_size：模型Batch Size。

- max_seqlen：模型输入长度。子目录的名字，比如，加载的模型尺寸是512，batch size 1
```shell
{
        "batch_size": 1,
        "max_seqlen": 512
}
```
IMAGE 表示使用的镜像名称。
- embedding_model_name：Embedding 模型的名称。

- embedding_GPUs: 运行 Embedding系列模型的GPU ID列表。列表数 = embedding_instance_nums。

- embedding_model_len：Embedding 系列模型输入长度。仅 Embedding 系列模型需要设置。

- embedding_instance_nums： 运行 Embedding 系列模型的实例数量。

- rerank_model_name：Rerank 模型的名称。

- rerank_GPUs：运行 Rerank 系列模型的GPU ID列表。列表数 = rerank_instance_nums。

- rerank_instance_nums： 运行 Rerank 系列模型的实例数量。
这里的embedding_model_name，rerank_model_name 名字可以修改为自定义的。

**步骤 1.** 根据实际情况选择“example/emb-rerank/xxx.yaml”文件中, 修改.env 变量

其中，“xxx”为embedding、reranker，请根据实际情况替换。

- embedding：表示启动 Embedding 系列模型服务。

- reranker：表示启动 Rerank 系列模型服务。

**步骤 2.**  启动模型服务。

```shell
cd /home/username/example/emb-rerank
docker-compose -f xxx.yaml up -d 
```

**步骤 3.** 检查模型服务是否启动成功。

- 如果模型为 Embedding 系列模式，则执行如下步骤。

1. 修改“example/emb-rerank/test/emb.py”高亮内容，分别将其修改为模型服务地址和模型名称。

> 模型服务地址格式为[http://IP:Port/v1/embeddings](http://IP:Port/v1/embeddings)。其中，IP为 模型服务IP地址，请根据实际情况设置。“Port”为模型服务端口,可在“example/emb-rerank/xxx.yaml”中查看“ports”参数的值确认其端口号。

> 模型名称需与“xxx.yaml”中的“model_name”保持一致。

```{code-block}
import requests

# 定义请求参数
response = requests.post(
    "http://localhost:9998/v1/embeddings",  # Embedding 端点
    json={
        "model": "emb_vacc",  # 替换为你的 Embedding 模型 UID（如 'bge-m3'）
        "input": "A man is eating pasta."    # 支持字符串或字符串列表
    }
)
print("Emb 结果:", response.json())
```

2. 执行测试脚本。
```shell
cd /home/username/example/emb-rerank/test
python3 emb.py
```

3. 执行测试脚本。
```shell
cd /home/username/example/emb-rerank/benchmark
python3 emb_concurrency.py
```

- 如果模型为 Rerank 系列模式，则执行如下步骤。

1. 修改“example/emb-rerank/test/rerank.py”高亮内容，将其分别修改为模型服务地址和模型名称。

> 模型服务地址格式为[http://IP:Port/v1/rerank](http://IP:Port/v1/rerank)。其中，IP为 模型服务IP地址，请根据实际情况设置。“Port”为模型服务端口,可在“example/emb-rerank/xxx.yaml”中查看“ports”参数的值确认其端口号。

> 模型名称需与“xxx.yaml”中的“model_name”保持一致。

```{code-block}
import requests
response = requests.post(
    "http://localhost:9999/v1/rerank",
    json={
        "model": "rerank_vacc",
        "query": "A man is eating pasta.",
        "documents": [
            "A man is eating food.",
            "A man is eating a piece of bread.",
            "The girl is carrying a baby.",
            "A man is riding a horse.",
            "A woman is playing violin."
        ],
        "return_documents": True
    }
)
print("Rerank 结果:", response.json())

```

2. 执行测试脚本。
```shell
cd /home/username/example/emb-rerank/test
python3 rerank.py
```
3. 执行并发测试脚本。
```shell
cd /home/username/example/emb-rerank/benchmark
python3 rerank_concurrency.py
```
### webui

- 浏览器输入 `http://${supervisor_host}:port`
- 通过 `Cluster Information` 页面查看集群信息
- 通过 `Running Models` 页面查看启动的模型
- `curl 'http://localhost:port/v1/models'`

![](./docs/imgs/launch_model.png)
> Note: deepseek v3/r1 当前的配置 tp 必须为32， max_model_len 必须小于等于 65536
> Note: vsx 加载VACC Embedding/Rerank模型， 每个模型建议用一个core



