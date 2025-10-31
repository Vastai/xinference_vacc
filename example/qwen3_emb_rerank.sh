#!/bin/bash

set -e  # 遇到错误立即退出
echo "=== 启动 Embedding 和 Rerank 服务 ==="
cd emb-rerank

echo "环境配置:"
cat .env
echo ""

docker-compose -f both.yaml up -d

echo "vacc-emb-rerank 服务容器正在启动中(请耐心等待)..."
echo "emb_vacc:端口9998"
echo "rerank_vacc:端口9999"
cd ..
# 设置默认参数
DEFAULT_TP_CONFIG="tp4"
DEFAULT_MODEL_NAME="Qwen3-30B-A3B-Instruct-2507-FP8"

# 显示用法说明
usage() {
    echo "用法: $0 [tp2|tp4] [模型名称]"
    echo "参数说明:"
    echo "  第一个参数: tp2 或 tp4 (默认: $DEFAULT_TP_CONFIG)"
    echo "  第二个参数: 模型名称 (默认: $DEFAULT_MODEL_NAME)"
    echo "可用模型:"
    echo "  - Qwen3-30B-A3B-FP8"
    echo "  - Qwen3-30B-A3B-Instruct-2507-FP8"
    echo "  - Qwen3-30B-A3B-Thinking-2507-FP8"
    echo ""
    echo "示例:"
    echo "  $0 tp2 Qwen3-30B-A3B-Thinking-2507-FP8"
    echo "  $0 tp4"
    echo "  $0 Qwen3-30B-A3B-FP8"
    exit 1
}

# 解析参数
TP_CONFIG="$DEFAULT_TP_CONFIG"
MODEL_NAME="$DEFAULT_MODEL_NAME"

# 处理参数
for arg in "$@"; do
    case "$arg" in
        tp2|tp4)
            TP_CONFIG="$arg"
            ;;
        Qwen3-30B-A3B-FP8|Qwen3-30B-A3B-Instruct-2507-FP8|Qwen3-30B-A3B-Thinking-2507-FP8)
            MODEL_NAME="$arg"
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "错误: 未知参数 '$arg'"
            usage
            ;;
    esac
done

# 验证参数
if [[ "$TP_CONFIG" != "tp2" && "$TP_CONFIG" != "tp4" ]]; then
    echo "错误: TP配置必须是 tp2 或 tp4"
    usage
fi

# 定义模型目录路径
MODEL_DIR="qwen3/$MODEL_NAME"

echo "=== 准备启动大模型服务 ==="
echo "配置: TP$TP_CONFIG, 模型: $MODEL_NAME"

# 检查必要文件
if [ ! -f "$MODEL_DIR/.env" ]; then
    echo "错误: 找不到 $MODEL_NAME 环境配置文件"
    echo "请检查路径: $MODEL_DIR/.env"
    exit 1
fi

if [ ! -f "emb-rerank/.env" ]; then
    echo "错误: 找不到 emb-rerank 环境配置文件"
    exit 1
fi

echo "=== 启动 $MODEL_NAME 大模型 ==="
cd "$MODEL_DIR"

echo "环境配置:"
cat .env
echo ""

# 根据参数选择 tp2.yaml 或 tp4.yaml
COMPOSE_FILE="${TP_CONFIG}.yaml"

if [ ! -f "$COMPOSE_FILE" ]; then
    echo "错误: 找不到 Docker Compose 文件 $COMPOSE_FILE"
    echo "当前目录: $(pwd)"
    echo "可用文件:"
    ls -la *.yaml 2>/dev/null || echo "没有找到 yaml 文件"
    exit 1
fi

echo "使用 $TP_CONFIG 配置: $COMPOSE_FILE"
docker-compose -f $COMPOSE_FILE up -d

echo "大模型服务容器正在启动中（请耐心等待）..."

echo ""
echo "=== 启动命令执行完成 ==="
echo "大模型使用的配置:"
echo "  - Tensor Parallel: $TP_CONFIG"
echo "  - 模型: $MODEL_NAME"
echo ""
echo "可以使用以下命令检查服务状态:"
echo "  docker-compose -f $MODEL_DIR/$COMPOSE_FILE ps"
echo "  docker-compose -f emb-rerank/both.yaml ps"