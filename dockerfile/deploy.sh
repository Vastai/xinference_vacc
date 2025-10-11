#!/bin/bash

set -e  # 遇到错误立即退出

echo "=== 开始自动化部署 ==="

# 检测系统架构
detect_architecture() {
    ARCH=$(uname -m)
    case $ARCH in
        x86_64)
            echo "检测到 x86_64 架构"
            DOCKER_IMAGE="harbor.vastaitech.com/ai_deliver/vllm_vacc:AI3.0_SP9_0811"
            DOCKERFILE_DIR="x86"
            MODEL_PACKAGE="emb_models-x86.tar.gz"
            ;;
        aarch64|arm64)
            echo "检测到 ARM64 架构"
            DOCKER_IMAGE="harbor.vastaitech.com/ai_deliver/vllm_vacc:AI3.0_SP9_0811_arm"
            DOCKERFILE_DIR="arm"
            MODEL_PACKAGE="emb_models-arm.tar.gz"
            ;;
        *)
            echo "不支持的架构: $ARCH"
            exit 1
            ;;
    esac
}

# 执行架构检测
detect_architecture

# 检查并安装 curl
if ! command -v curl &> /dev/null; then
    echo "curl 未安装，开始安装..."
    
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        case $ID in
            ubuntu|debian)
                sudo apt update
                sudo apt install -y curl
                ;;
            centos|rhel|fedora)
                sudo yum install -y curl
                ;;
            *)
                echo "不支持的Linux发行版，请手动安装curl"
                exit 1
                ;;
        esac
    else
        echo "无法检测系统类型，请手动安装curl"
        exit 1
    fi
    echo "✅ curl 安装完成"
else
    echo "✅ curl 已安装"
fi

# 检查并安装 Git
if ! command -v git &> /dev/null; then
    echo "Git 未安装，开始安装..."
    
    # 检测系统类型并安装 Git
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        case $ID in
            ubuntu|debian)
                sudo apt update
                sudo apt install -y git
                ;;
            centos|rhel|fedora)
                sudo yum install -y git
                ;;
            *)
                echo "不支持的Linux发行版，请手动安装Git"
                exit 1
                ;;
        esac
    else
        echo "无法检测系统类型，请手动安装Git"
        exit 1
    fi
fi

# 验证 Git 安装
echo "验证 Git 安装:"
git --version

# 克隆 xinference_vacc 仓库
echo "克隆 xinference_vacc 仓库..."
if [ ! -d "xinference_vacc" ]; then
    git clone https://github.com/Vastai/xinference_vacc.git
else
    echo "xinference_vacc 目录已存在，跳过克隆"
    cd xinference_vacc
    git pull  # 更新到最新代码
    cd ..
fi

# 检查并安装 Docker
if ! command -v docker &> /dev/null; then
    echo "Docker 未安装，开始安装..."
    
    # 安装 Docker
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    
    # 添加当前用户到 docker 组
    sudo usermod -aG docker $USER
    newgrp docker
    
    # 启动 Docker 服务
    sudo systemctl start docker
    sudo systemctl enable docker
fi

# 检查并安装 Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose 未安装，开始安装..."
    
    # 安装 Docker Compose
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
    
    # 创建符号链接
    sudo ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
fi

# 检查并安装 make 工具
if ! command -v make &> /dev/null; then
    echo "make 工具未安装，开始安装..."
    
    # 检测系统类型并安装 make
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        case $ID in
            ubuntu|debian)
                sudo apt update
                sudo apt install -y make
                ;;
            centos|rhel|fedora)
                sudo yum install -y make
                ;;
            alpine)
                sudo apk add make
                ;;
            *)
                echo "不支持的Linux发行版，请手动安装make"
                exit 1
                ;;
        esac
    else
        echo "无法检测系统类型，请手动安装make"
        exit 1
    fi
    echo "✅ make 工具安装完成"
else
    echo "✅ make 工具已安装"
fi

# 验证 make 安装
echo "验证 make 版本:"
make --version

# 检查并安装编译工具链
install_build_essentials() {
    echo "检查编译工具链..."
    
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        case $ID in
            ubuntu|debian)
                if ! dpkg -l | grep -q "build-essential"; then
                    echo "安装 build-essential..."
                    sudo apt update
                    sudo apt install -y build-essential
                fi
                if ! command -v gcc &> /dev/null; then
                    echo "安装 gcc..."
                    sudo apt install -y gcc
                fi
                if ! command -v g++ &> /dev/null; then
                    echo "安装 g++..."
                    sudo apt install -y g++
                fi
                ;;
            centos|rhel|fedora)
                if ! rpm -q gcc-c++ &> /dev/null; then
                    echo "安装开发工具..."
                    sudo yum groupinstall -y "Development Tools"
                    sudo yum install -y gcc-c++
                fi
                ;;
            alpine)
                echo "安装 Alpine 开发工具..."
                sudo apk add build-base
                ;;
            *)
                echo "不支持的Linux发行版，请手动安装编译工具"
                return 1
                ;;
        esac
    else
        echo "无法检测系统类型"
        return 1
    fi
    
    # 验证安装
    if command -v gcc &> /dev/null && command -v g++ &> /dev/null; then
        echo "✅ 编译工具链安装完成"
        echo "GCC 版本: $(gcc --version | head -n1)"
        echo "G++ 版本: $(g++ --version | head -n1)"
        return 0
    else
        echo "❌ 编译工具链安装失败"
        return 1
    fi
}

# 安装编译工具链
install_build_essentials

# 验证安装
echo "验证 Docker 和 Docker Compose 安装:"
docker --version
docker-compose --version

# 拉取镜像（根据架构选择不同的镜像）
echo "拉取 Docker 镜像: $DOCKER_IMAGE"
docker pull $DOCKER_IMAGE

# 进入项目目录
cd xinference_vacc
echo "开始轮子包编译.."
python3 setup.py bdist_wheel
# 编译好轮子包，在dist目录下
cp dist/xinf*.whl dockerfile/$DOCKERFILE_DIR
echo "完成轮子包编译.."
cd dockerfile/$DOCKERFILE_DIR

# 下载必要的文件
# echo "下载驱动..."
# wget -O vastai_driver_install_d3_3_v2_7_a3_0_9c31939_00.25.08.11.run https://devops.vastai.com/kapis/artifact.kubesphere.io/v1alpha1/artifact?artifactid=6558 --no-check-certificate

# # 安装驱动（在宿主机上安装）
# echo "安装驱动..."
# chmod +x vastai_driver_install_d3_3_v2_7_a3_0_9c31939_00.25.08.11.run
# sudo bash vastai_driver_install_d3_3_v2_7_a3_0_9c31939_00.25.08.11.run install

echo "构建 Docker 镜像..."
docker build -t harbor.vastaitech.com/ai_deliver/xinference_vacc_151:AI3.0_SP9_0811 .

## 进入 example 目录
cd ../../example/emb-rerank

## 下载 emb/rerank 模型
echo "下载emb rerank 模型: $MODEL_PACKAGE"
# 根据架构选择不同的模型包URL
# if [ "$ARCH" = "x86_64" ]; then
#     wget http://ce-support.vastaitech.com/ytcx/emb_models-x86.tar.gz
# elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
#     wget http://ce-support.vastaitech.com/ytcx/emb_models-arm.tar.gz
# fi

# echo "完成下载, 开始解压"
# tar -zxvf emb_models*.tar.gz
echo "完成解压"

## 获取当前目录，并修改 .env 文件中的 EMB_DATA_DIR 和 RERANK_DATA_DIR 变量
echo "更新 .env 文件中的路径..."
CURRENT_DIR=$(pwd)

# 备份原文件
if [ -f .env ]; then
    cp .env .env.backup
    echo "已备份原 .env 文件为 .env.backup"
fi

# 更新 .env 文件中的路径
if [ -f .env ]; then
    # 方法1：使用 sed 替换
    sed -i "s|/home/tonyguo|${CURRENT_DIR}|g" .env
else
    # 如果 .env 文件不存在，创建它
    cat > .env << EOF
# 模型目录的路径
EMB_DATA_DIR=${CURRENT_DIR}/emb_models/bge-m3-vacc
RERANK_DATA_DIR=${CURRENT_DIR}/emb_models/bge-reranker-v2-m3-vacc
EOF
fi

# 显示修改后的内容
echo "更新后的 .env 文件内容："
cat .env

echo "启动容器..."
docker-compose -f both.yaml up -d

echo "等待模型加载完成信号（由于每个实例要加载多个尺寸的模型，请耐心等待）..."

# 静默等待，只显示开始和结束
if timeout 1200 bash -c '
    while ! docker-compose -f both.yaml logs | grep -q "模型加载完成"; do
        sleep 30
    done
'; then
    echo "✅ 模型加载完成！"
else
    echo "⚠️  等待超时，但继续执行测试..."
fi

echo "执行测试..."
cd test
if [ -f "emb.sh" ]; then
    bash emb.sh
    echo "emb测试完成"
else
    echo "emb测试脚本不存在"
fi

if [ -f "rerank.sh" ]; then
    bash rerank.sh
    echo "rerank测试完成"
else
    echo "rerank测试脚本不存在"
fi

echo "=== 部署完成 ==="

