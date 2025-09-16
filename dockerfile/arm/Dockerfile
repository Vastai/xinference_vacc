FROM harbor.vastaitech.com/ai_deliver/vllm_vacc:AI3.0_SP9_0811_arm

# 设置工作目录
WORKDIR /test

# 将当前目录下的所有wheel文件拷贝到容器的/test目录下, 假设当前目录已经有vaststreamx/xinference_vacc 的轮子包
RUN wget -O ai-v3.0.0-20250811-1066-linux-aarch64-sdk2-python3.12.bin  http://devops.vastai.com/kapis/artifact.kubesphere.io/v1alpha1/artifact?artifactid=6553 --no-check-certificate
COPY *.whl /test/

# 设置 pip 使用清华源（永久生效）
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install xinference==1.5.1
# 安装指定的vaststreamx wheel包
RUN pip install /test/vaststreamx*.whl
RUN pip install /test/xinference_vacc-1.5.1-py3-none-any.whl
RUN chmod +x /test/ai-v3.0.0-20250811-1066-linux-aarch64-sdk2-python3.12.bin
RUN /test/ai-v3.0.0-20250811-1066-linux-aarch64-sdk2-python3.12.bin
## 拷贝启动脚本
COPY scripts /test/
# 可以添加其他需要的命令或配置
# ...

# 设置容器启动时的默认命令（根据你的需求修改）
CMD ["bash"]
