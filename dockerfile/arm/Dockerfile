FROM harbor.vastaitech.com/ai_deliver/vllm_vacc:AI3.1_POC_1013_arm

# 设置工作目录
WORKDIR /test

# 将当前目录下的所有wheel文件拷贝到容器的/test目录下, 假设当前目录已经有vaststreamx/xinference_vacc 的轮子包
RUN wget -O ai-v3.1.0-20251013-1914-linux-aarch64-sdk2-python3.12.bin  http://devops.vastai.com/kapis/artifact.kubesphere.io/v1alpha1/artifact?artifactid=6553 --no-check-certificate
RUN wget -O vaststreamx-2.8.8-cp312-cp312-linux_aarch64.whl  https://devops.vastai.com/kapis/artifact.kubesphere.io/v1alpha1/artifact?artifactid=7244 --no-check-certificate
COPY *.whl /test/

# 设置 pip 使用清华源（永久生效）
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 安装指定的vaststreamx wheel包
RUN chmod +x /test/ai*.bin
RUN /test/ai*.bin
RUN pip install /test/vaststreamx*.whl
RUN pip install xinference==1.5.1
RUN pip install /test/xinference_vacc-1.5.1-py3-none-any.whl

## 拷贝启动脚本
COPY scripts /test/
# 可以添加其他需要的命令或配置
# ...

# 设置容器启动时的默认命令（根据你的需求修改）
CMD ["bash"]
