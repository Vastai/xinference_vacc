<div id=top align="center">

![logo](./images/index/logo.png)
[![License](https://img.shields.io/badge/license-Apache_2.0-yellow)](LICENSE)
[![company](https://img.shields.io/badge/vastaitech.com-blue)](https://www.vastaitech.com/)


</div>

---
# xinference on vacc
# 简介


Xinference（Xorbits Inference）是一个性能强大且功能全面的开源分布式推理框架，旨在简化本地和云端模型的部署与管理，支持多样化的模型类型（如大语言模型、嵌入模型和多模态模型）和硬件加速（支持CPU、GPU等）。Xinference不仅可以在本地运行推理模型，还支持在分布式集群环境下高效推理，可以轻松扩展以处理更大规模的推理任务。


xinference 目前适配了瀚博硬件，支持使用瀚博硬件设备进行LLM系列、Embedding系列、Rerank系列, VL系列模型的部署和推理。
具体PR 如下：
https://github.com/xorbitsai/inference/pull/4382  

https://github.com/xorbitsai/inference/pull/4385  

## Engine
- [`20251219`]: suppport vLLM engine

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
## 准备镜像（确保有外网，直接下载公开镜像） 
x86平台：
docker pull harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-25.12.SP1  
arm平台：
docker pull harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-25.12.SP1_arm
## 准备模型
根据您的需要，准备好模型。下载到服务器。
```
example目录结构如下所示
├── ds3
│   ├── cluster
│   ├── MTP
│   ├── nonMTP
│   └── test
├── text2vec
│   ├── bge
│   ├── qwen3
├── hybrid
│   ├── ds_text2vec
│   └── qwen3_text2vec
├── install_docker_compose.sh
└── qwen3
    ├── hybrid
    ├── instruct
    ├── thinking
    └── test
```

目录/文件	说明
| 目录 | 意义| 
|-------|-------|
| ds3   | DeepSeek-V3 或 DeepSeek-R1 系列模型服务的 Docker Compose 文件及测试脚本。|
| ds3/cluster | （可选）用于跨机器启动 DeepSeek 模型的配置。 | 
| ds3/MTP | 包含引入多令牌预测技术 (Multi-Token Prediction) 的配置，用于提升模型推理性能。 |
| ds3/nonMTP | 未引入多令牌预测技术 (MTP) 的标准部署配置。|
| ds3/test | python/curl 测试ds服务是否通 |
| text2vec | 启动 Embedding 或 Rerank 系列模型服务的 Docker Compose 文件及测试脚本。|
| text2vec/bge | 启动bge系列text2vec模型的Docker Compose配置文件。|
| text2vec/qwen3 | 启动qwen3系列text2vec模型的Docker Compose配置文件。|
| qwen3      | 启动 Qwen3 系列模型服务的 Docker Compose 文件及测试脚本。|
| qwen3/hybrid      | 启动hybrid系列模型的Docker Compose 配置文件|
| qwen3/instruct      | 启动instruct系列模型的Docker Compose 配置文件|
| qwen3/thinking      | 启动thinking系列模型的Docker Compose 配置文件|
| qwen3/test      | python/curl测试模型服务是否通|
| hybrid    | 同时部署LLM和text2vec 模型|
| hybrid/ds_text2vec   | 同时部署DS 系列LLM 和text2vec 模型|
| hybrid/qwen3_text2vec   | 同时部署qwen3 系列LLM 和text2vec 模型|

| 文件 | 意义 |
|-------|-------|
| tp2.yaml | 部署 Qwen3-30B系列模型实例的 Docker Compose 配置文件, TP2。 |
| tp4.yaml        | 部署 Qwen3-30B系列模型实例的 Docker Compose 配置文件, TP4。 |
| tp16.yaml | 部署 Qwen3-235B系列模型实例的 Docker Compose 配置文件, TP16。 |
| tp16_100k.yaml | 部署 Qwen3-235B系列模型实例的 Docker Compose 配置文件，最大支持输入100K, TP16。data pipeline size 1|
| tp32.yaml | 部署 DS系列模型实例的 Docker Compose 配置文件, TP32。 |
| tp32_100k.yaml | 部署 DS系列模型实例的 Docker Compose 配置文件，非MTP模式，最大支持输入100K, TP32。data pipeline size 2|

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

通过 xinference_vacc 启动 DeepSeek-V3.1 或者 DeepSeek-V3 或 DeepSeek-R1系列模型，其步骤如下所示。

**前提条件**

example/ds3 的每个子目录下，都有.env 变量
用于配置yaml 中的变量。下面以DeepSeek-V3.1 举例说明。
```shell
# 模型目录的路径
# 新增服务端口变量
service_port=9997
#表示存放模型目录的路径。  
LLM_DATA_DIR=/logs
# 镜像设置
IMAGE=harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-25.12.SP1
## 如果是arm 平台，公版的镜像是 harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-25.12.SP1_arm
# 参数设置
model_name=DeepSeek-V3.1
#来指定具体模型目录。  
model_directory=DeepSeek-V3.1
#表示在哪些GPU IDS 上面加载，目前ds 仅支持tp32。
# GPU_LIST: GPU ID列表。列表数= TP32 * instance_nums。
GPU_LIST=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
##需要启动的模型实例个数
instance_nums=1
```  
这里要注意的是模型名字。  
| 模型名字（不可更改） | 模型目录| 
|-------|-------|
| deepseek-v3 | DeepSeek-V3、DeepSeek-V3-0324 | 
| deepseek-r1 | DeepSeek-R1、DeepSeek-R1-0528 |
| DeepSeek-V3.1 |DeepSeek-V3.1-Terminus、DeepSeek-V3.1 |


**步骤 1.** 
根据实际情况选择“example/ds3/MTP(或者nonMTP)/xxx.yaml”文件, 并修改.env

“xxx”为模型名，请根据实际情况替换。

**步骤 2.**  启动模型服务。

```shell
cd /home/username/example/ds3/MTP(或者nonMTP)
docker-compose -f xxx.yaml up -d 
```

**步骤 3.** 检查模型服务是否启动成功。

- 如果模型为 DeepSeek-V3系列模型，按如下步骤执行。


1. 修改“example/ds3/test/31chat.py”中“base_url”。

> “base_url”为 模型服务地址，格式为[http://IP:Port/v1](http://IP:Port/v1)。IP为 模型服务IP地址，请根据实际情况设置。“Port”为模型服务端口,可在“example/ds3/xxx.yaml”中查看“ports”参数的值确认其端口号。



```{code-block}
from openai import OpenAI
client = OpenAI(base_url="http://localhost:9997/v1", api_key="EMPTY")

response = client.chat.completions.create(
  model="DeepSeek-V3.1",
  messages=[{"role": "user", "content": "中国直辖市是哪里"}],
  temperature=0.5,
)
print(response.choices[0].message.content)
```

2. 执行测试脚本。
```shell
cd /home/username/example/ds3/test
python3 31chat.py
```



- 如果模型为 DeepSeek-R1系列模型，则执行如下步骤执行。

1. 修改“example/ds3/test/r1chat.py”中“base_url”。

> “base_url”为 模型服务地址，格式为[http://IP:Port/v1](http://IP:Port/v1)。IP为 模型服务IP地址，请根据实际情况设置。“Port”为模型服务端口，可在“example/ds3/xxx.yaml”中查看“ports”参数的值确认其端口号。



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
cd /home/username/example/ds3/test
python3 r1chat.py
```

## 跨机启动 DS3 系列模型服务
**（可选，根据您手上的资源来）**

	如果您手上有两台或者更多瀚博的一体机，并且网络能互通，而且您这边有多服务需求。
由于我们每台一体机最多只能部署两个DeepSeek 模型，您这边可以根据需要搭建集群。  
  

**前提条件**  

根据实际情况修改“example/ds3/cluster/*.yaml”文件中“volumes”参数，将其修改为实际模型权重文件夹所在路径。  
注意，多台机器的模型在物理机的绝对路径需要一致，才能跨机加载，这边建议可以用网盘。  

在cluster 目录下，是一个场景例子。这边做一下解释说明，可以根据您那边需要修改。  

场景：
	假设我们有两台机器，分别是10.24.73.25/10.24.73.23, 每台机器都满足条件  
	（镜像一致，模型已经准备好，16张VA16）  

我们想要加载4个Deepseek-V3.1 模型服务，并通过一个supervisor入口来调度请求。  

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
用于配置yaml 中的变量。下面以Qwen3-30B-A3B-FP8举例
```shell
# 模型目录的路径
# 新增服务端口变量
service_port=9998
LLM_DATA_DIR=/logs
# 镜像设置
IMAGE=harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-25.12.SP1
## 如果是arm 平台，公版的镜像是 harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-25.12.SP1_arm
# 参数设置
model_name=qwen3
model_directory=Qwen3-30B-A3B-FP8
GPU_LIST=16,17,18,19
instance_nums=2
```
- GPU_LIST: GPU ID列表。列表数= TP * instance_nums。 
- instance_nums：实例数量。 
针对 Qwen3-30B-A3B 系列模型，当前TP仅支持 2 或 4 。  
如果是针对 Qwen3-235B-A3B，当前TP仅支持16。  

例如，TP=2，instance_nums=2，列表数= 2 * instance_nums，可设置为 0,1,2,3。  

如果是TP=4， instance_nums=2，列表数= 2 * instance_nums，可设置为 0,1,2,3,4,5,6,7。  

如果是TP=16，instance_nums=1, 列表数= 1 * instance_nums，可设置为 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15。


| 模型名字（不可更改） | 模型目录|
|-------|-------|
| qwen3 | Qwen3-30B-A3B-FP8、Qwen3-30B-A3B-GPTQ-Int4、 Qwen3-235B-A22B-FP8|
| Qwen3-Instruct | Qwen3-30B-A3B-Instruct-2507-FP8、Qwen3-235B-A22B-Instruct-2507 |
| Qwen3-Thinking | Qwen3-30B-A3B-Thinking-2507-FP8, Qwen3-235B-A22B-Thinking-2507 |


**步骤 1.** 根据实际情况选择“example/qwen3/model_type/model_name/xxx.yaml”文件,
并修改.env 文件

“model_name”为模型名称，“xxx”为tp2 或 tp4，请根据实际情况替换。



**步骤 2.**  启动模型服务。

```shell
cd /home/username/example/qwen3/model_type/model_name
docker-compose -f xxx.yaml up -d 
```

**步骤 3.** 检查模型服务是否启动成功。

1. 修改“example/qwen3/test/model_type/chat.py”中“base_url”。
> “base_url”为 模型服务地址，格式为[http://IP:Port/v1](http://IP:Port/v1)。IP为 模型服务IP地址，请根据实际情况设置。“Port”为模型服务端口,可在“example/qwen3/xxx.yaml”中查看“ports”参数的值确认其端口号。

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

## 启动 Qwen3vl 系列模型服务


通过 xinference_vacc 启动 Qwen3vl 系列模型，其步骤如下所示。  

**前提条件**  
准备 Qwen3-VL-30B-A3B-Thinking-FP8 或者Qwen3-VL-30B-A3B-Instruct-FP8原始模型。Qwen3-VL-30B支持tp4 模式。  

目前支持的情况如下：

| 模型名字（不可更改） | 模型目录|
|-------|-------|
| Qwen3-VL-Instruct | Qwen3-VL-30B-A3B-Instruct-FP8 |
| Qwen3-VL-Thinking | Qwen3-VL-30B-A3B-Thinking-FP8 |

下面以Qwen3-VL-Thinking 举例说明  

example/qwen3/instruct/Qwen3-VL-Instruct目录下，有.env 变量
用于配置yaml 中的变量。

```shell
# 模型目录的路径
# 新增服务端口变量
service_port=9998
LLM_DATA_DIR=/logs
# 镜像设置
IMAGE=harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-25.12.SP1
## 如果是arm 平台，公版的镜像是 harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-25.12.SP1_arm
# 参数设置
model_name=Qwen3-VL-Thinking
model_directory=Qwen3-VL-30B-A3B-Thinking-FP8
GPU_LIST=0,1,2,3
instance_nums=1
```
- GPU_LIST: GPU ID列表。列表数= TP4 * instance_nums。 
假如 instance_nums=2，列表数= tp4 * instance_nums，可设置为 0,1,2,3,4,5,6,7。   
- instance_nums：实例数量。  

**步骤 1.** 根据实际情况选择“example/qwen3/model_type/model_name/xxx.yaml”文件,
并修改.env 文件

“model_name”为模型名称，“xxx”为tp2 或 tp4，请根据实际情况替换。

针对 Qwen3vl 系列模型，当前TP仅支持 2 或 4 。

**步骤 2.**  启动模型服务。

```shell
cd /home/username/example/qwen3/model_type/model_name
docker-compose -f xxx.yaml up -d 
```

**步骤 3.** 检查模型服务是否启动成功。

1. 执行测试脚本。
```shell
cd /home/username/example/qwen3/test/vl
python vl.py --port 9993 --model Qwen3-VL-Thinking --prompt "描述图片" --image test.jpg
端口，模型，prompt 可以根据需要配置。
```

## 启动 Embedding 或 Rereank 系列模型服务

通过 xinference_vacc 启动 Embedding 或 Rerank 系列模型，其步骤如下所示。  
目前的支持情况如下：
| 模型名字（不可更改） | 模型目录|
|-------|-------|
| Qwen3-Embedding-0.6B| Qwen3-Embedding-0.6B|
| Qwen3-Reranker-0.6B | Qwen3-Reranker-0.6B |
| bge-m3 | bge-m3 |
| bge-reranker-v2-m3 | bge-reranker-v2-m3 |

**前提条件**  
准备好需要的模型。  

example/text2vec 下面有 bge 或者qwen3 子目录，都有.env 变量， 用于配置yaml 中的变量。  

我们以qwen3 类型的text2vec 为例子。
举例说明，进到text2vec/qwen3目录下的.env
```shell
# 模型目录的路径
# 新增服务端口变量
service_port=9996
EMB_DATA_DIR=/disk/models/
RERANK_DATA_DIR=/disk/models/
# 镜像设置
IMAGE=harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-25.12.SP1
## 如果是arm 平台，公版的镜像是 harbor.vastaitech.com/ai_deliver/xinference_vacc:VVI-25.12.SP1_arm
# 参数设置
embedding_model_name=Qwen3-Embedding-0.6B
embedding_model_directory=Qwen3-Embedding-0.6B
embedding_tp=2 
##embedding_tp=1 /2/ 4
embedding_GPUs=8,9
embedding_instance_nums=1

rerank_model_name=Qwen3-Reranker-0.6B
rerank_model_directory=Qwen3-Reranker-0.6B
rerank_tp=2
##rerank_tp=1 /2/ 4
rerank_GPUs=10,11
rerank_instance_nums=1
```
EMB_DATA_DIR,RERANK_DATA_DIR表示emb/rerank模型目录的路径。

IMAGE 表示使用的镜像名称。
- embedding_model_name：Embedding 模型的名称。
- embedding_model_directory：Embedding 模型目录的名称。
- embedding_GPUs: 运行 Embedding系列模型的GPU ID列表。列表数 = embedding_tp *embedding_instance_nums 。
- embedding_instance_nums： 运行 Embedding 系列模型的实例数量。
- embedding_tp： 启动每个Embedding模型实例， 需要的tp 个数，目前支持tp1, tp2, tp4
- rerank_model_name：Rerank 模型的名称。
- rerank_model_directory：Embedding 模型目录的名称。
- rerank_GPUs：运行 Rerank 系列模型的GPU ID列表。列表数 = rerank_tp * rerank_instance_nums。
- rerank_tp： 启动每个Embedding模型实例， 需要的tp 个数，目前支持tp1, tp2, tp4
- rerank_instance_nums： 运行 Rerank 系列模型的实例数量。  

例如，embedding_tp=1，embedding_instance_nums=4，embedding_GPUs= 1 * 4，可设置为 0,1,2,3 。  

例如，rerank_tp=2，rerank_instance_nums=2，rerank_GPUs= 2 * 2，可设置为 0,1,2,3 。  

**步骤 1.** 根据实际情况选择“example/text2vec/qwen3/xxx.yaml”文件中, 修改.env 变量

“xxx”为embedding、reranker，请根据实际情况替换。
- both：表示同时启动 Embedding, Rerank 系列模型服务。

- embedding：表示启动 Embedding 系列模型服务。

- reranker：表示启动 Rerank 系列模型服务。

**步骤 2.**  启动模型服务。

```shell
cd /home/username/example/text2vec/qwen3
docker-compose -f xxx.yaml up -d 
```

**步骤 3.** 检查模型服务是否启动成功。

- 如果模型为 Embedding 系列模式，则执行如下步骤。

1. 修改“example/text2vec/qwen3/test/emb.py”高亮内容，分别将其修改为模型服务地址和模型名称。

> 模型服务地址格式为[http://IP:Port/v1/embeddings](http://IP:Port/v1/embeddings)。IP为 模型服务IP地址，请根据实际情况设置。“Port”为模型服务端口,可在“example/text2vec/xxx.yaml”中查看“ports”参数的值确认其端口号。

> 模型名称需与“xxx.yaml”中的“model_name”保持一致。

```{code-block}
import requests

# 定义请求参数
response = requests.post(
    "http://localhost:9998/v1/embeddings",  # Embedding 端点
    json={
        "model": "Qwen3-Embedding-0.6B",    # 模型名字（不可更改）
        "input": "A man is eating pasta."    # 支持字符串或字符串列表
    }
)
print("Emb 结果:", response.json())
```

2. 执行测试脚本。
```shell
cd /home/username/example/text2vec/qwen3/test
python3 emb.py
```

- 如果模型为 Rerank 系列模式，则执行如下步骤。

1. 修改“example/text2vec/qwen3/test/rerank.py”高亮内容，将其分别修改为模型服务地址和模型名称。

> 模型服务地址格式为[http://IP:Port/v1/rerank](http://IP:Port/v1/rerank)。IP为 模型服务IP地址，请根据实际情况设置。“Port”为模型服务端口,可在“example/text2vec/xxx.yaml”中查看“ports”参数的值确认其端口号。

> 模型名称需与“xxx.yaml”中的“model_name”保持一致。

```{code-block}
import requests
response = requests.post(
    "http://localhost:9999/v1/rerank",
    json={
        "model": "Qwen3-Reranker-0.6B",
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
cd /home/username/example/text2vec/qwen3/test
python3 rerank.py
```
### webui

- 浏览器输入 `http://${supervisor_host}:port`
- 通过 `Cluster Information` 页面查看集群信息
- 通过 `Running Models` 页面查看启动的模型
- `curl 'http://localhost:port/v1/models'`


模型最大上下文长度限制：

针对 DeepSeek-V3/R1/V3.1 系列模型，模型最大上下文长度为 64K。  

针对 Qwen3 系列模型，如果 TP 为 2，模型最大上下文长度为 64K；如果TP 为 4或16，模型最大上下文长度为 128K。  

针对 Embedding/Rerank模型， bge-m3, bge-reranker-v2-m3 默认启动最大长度8192。  

Qwen3-Embedding-0.6B 最大长度 65536, Qwen3-Rerank-0.6B 默认最大长度 40960
> Note:
单模型同时支持最大并发数为 4。如果有多并发需求，可以用多副本
对于超出上下文长度的请求，服务端会拦截不做处理，客户端需自行校验请求长度。

