# 下载最新版本（替换为最新版本号）
DOCKER_COMPOSE_VERSION=v2.26.1
sudo curl -L "https://github.com/docker/compose/releases/download/${DOCKER_COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

# 添加执行权限
sudo chmod +x /usr/local/bin/docker-compose
sudo cp /usr/local/bin/docker-compose /usr/bin/docker-compose

# 验证安装
docker-compose --version
# 应该输出类似：Docker Compose version v2.26.1
