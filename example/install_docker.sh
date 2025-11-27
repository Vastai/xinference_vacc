tar -zxvf docker-28.1.1.tgz
sudo mkdir -p /usr/local/docker/bin
sudo mkdir -p /usr/bin/docker
cd docker
sudo cp docker* containerd* ctr runc /usr/local/docker/bin/
sudo cp docker dockerd containerd containerd-shim-runc-v2 docker-init docker-proxy runc /usr/bin/
sudo chmod +x /usr/bin/docker /usr/bin/dockerd /usr/bin/containerd /usr/bin/containerd-shim-runc-v2 /usr/bin/docker-init /usr/bin/docker-proxy /usr/bin/runc
sudo tee /etc/systemd/system/docker.service > /dev/null <<EOF
[Unit]
Description=Docker Application Container Engine
Documentation=https://docs.docker.com
After=network-online.target firewalld.service containerd.service
Wants=network-online.target
Requires=containerd.service

[Service]
Type=notify
ExecStart=/usr/bin/dockerd
ExecReload=/bin/kill -s HUP \$MAINPID
TimeoutSec=0
RestartSec=2
Restart=always
StartLimitBurst=3
StartLimitInterval=60s
LimitNOFILE=infinity
LimitNPROC=infinity
LimitCORE=infinity
TasksMax=infinity
Delegate=yes
KillMode=process

[Install]
WantedBy=multi-user.target
EOF

sudo tee /etc/systemd/system/containerd.service > /dev/null <<EOF
[Unit]
Description=containerd container runtime
Documentation=https://containerd.io
After=network.target local-fs.target

[Service]
ExecStartPre=-/sbin/modprobe overlay
ExecStart=/usr/bin/containerd
Type=notify
Delegate=yes
KillMode=process
Restart=always
RestartSec=5
LimitNOFILE=infinity
LimitNPROC=infinity
LimitCORE=infinity
TasksMax=infinity

[Install]
WantedBy=multi-user.target
EOF

# 创建配置目录
sudo mkdir -p /etc/docker
sudo mkdir -p /etc/containerd

# 生成默认的 docker daemon 配置
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "registry-mirrors": [
    "https://docker.mirrors.ustc.edu.cn",
    "https://hub-mirror.c.163.com"
  ],
  "data-root": "/var/lib/docker",
  "exec-opts": ["native.cgroupdriver=systemd"],
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m"
  },
  "storage-driver": "overlay2"
}
EOF

# 生成 containerd 配置
sudo /usr/bin/containerd config default | sudo tee /etc/containerd/config.toml > /dev/null

# 创建 Docker 数据目录
sudo mkdir -p /var/lib/docker
sudo mkdir -p /var/run/docker

# 创建 containerd 目录
sudo mkdir -p /var/lib/containerd
sudo mkdir -p /var/run/containerd

# 重新加载 systemd
sudo systemctl daemon-reload

# 启动 containerd
sudo systemctl enable containerd
sudo systemctl start containerd

# 启动 docker
sudo systemctl enable docker
sudo systemctl start docker

# 检查 Docker 版本
docker --version

# 检查 Docker 服务状态
sudo systemctl status docker

# 检查所有组件
docker info

# 将当前用户添加到 docker 组，避免每次使用 sudo
sudo usermod -aG docker $USER

# 重新登录或执行以下命令使分组生效
newgrp docker
