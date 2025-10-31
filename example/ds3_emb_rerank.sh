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
DEFAULT_MTP_MODE="MTP"
DEFAULT_MODEL_NAME="ds31"

# 显示用法说明
usage() {
    echo "用法: $0 [MTP|nonMTP] [模型名称]"
    echo "参数说明:"
    echo "  第一个参数: MTP 或 nonMTP (默认: $DEFAULT_MTP_MODE)"
    echo "  第二个参数: 模型名称 (默认: $DEFAULT_MODEL_NAME)"
    echo "可用模型:"
    echo "  - ds31"
    echo "  - dsr1"
    echo ""
    echo "示例:"
    echo "  $0 MTP ds31"
    echo "  $0 nonMTP dsr1"
    echo "  $0 ds31"
    exit 1
}

# 解析参数
MTP_MODE="$DEFAULT_MTP_MODE"
MODEL_NAME="$DEFAULT_MODEL_NAME"

# 处理参数
for arg in "$@"; do
    case "$arg" in
        MTP|nonMTP)
            MTP_MODE="$arg"
            ;;
        ds31|dsr1)
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
if [[ "$MTP_MODE" != "MTP" && "$MTP_MODE" != "nonMTP" ]]; then
    echo "错误: MTP模式必须是 MTP 或 nonMTP"
    usage
fi

if [[ "$MODEL_NAME" != "ds31" && "$MODEL_NAME" != "dsr1" ]]; then
    echo "错误: 模型名称必须是 ds31 或 dsr1"
    usage
fi

# 定义模型目录路径和配置文件
MODEL_DIR="ds3"
COMPOSE_FILE="$MTP_MODE/${MODEL_NAME}-1model.yaml"

echo "=== 准备启动大模型服务 ==="
echo "配置: $MTP_MODE, 模型: $MODEL_NAME"
echo "使用配置文件: $COMPOSE_FILE"

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

# 检查 Docker Compose 文件是否存在
if [ ! -f "$MODEL_DIR/$COMPOSE_FILE" ]; then
    echo "错误: 找不到 Docker Compose 文件 $COMPOSE_FILE"
    echo "当前目录: $(pwd)"
    echo "在 $MODEL_DIR 目录中查找以下文件:"
    ls -la $MODEL_DIR/$MTP_MODE/*.yaml 2>/dev/null || echo "没有找到对应的 yaml 文件"
    exit 1
fi

echo "=== 启动 $MODEL_NAME 大模型 ==="
cd "$MODEL_DIR"

echo "环境配置:"
cat .env
echo ""

echo "使用 $MTP_MODE 模式: $COMPOSE_FILE"
docker-compose -f $COMPOSE_FILE up -d

echo "大模型服务容器正在启动中（请耐心等待）..."

echo ""
echo "=== 启动命令执行完成 ==="
echo "大模型使用的配置:"
echo "  - MTP模式: $MTP_MODE"
echo "  - 模型: $MODEL_NAME"
echo "  - 配置文件: $COMPOSE_FILE"
echo ""
echo "可以使用以下命令检查服务状态:"
echo "  docker-compose -f $COMPOSE_FILE ps"
echo "  docker-compose -f ../emb-rerank/both.yaml ps"