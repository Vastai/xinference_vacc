FROM harbor.vastaitech.com/ai_deliver/vllm_vacc:VVI-25.12.SP2

# 设置工作目录
WORKDIR /test
# 设置 pip 使用清华源（永久生效）
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN git clone https://github.com/xorbitsai/inference.git
WORKDIR /test/inference
RUN git checkout 00957020f29ee8ffd918eec877833d7904966ff0
RUN pip install -e .[vllm]
RUN pip install xoscar==0.8.1
RUN pip install qwen_omni_utils

## 拷贝启动脚本
COPY scripts /test/
# 可以添加其他需要的命令或配置
# ...

# 设置容器启动时的默认命令（根据你的需求修改）
CMD ["bash"]
