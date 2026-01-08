FROM harbor.vastaitech.com/ai_deliver/vllm_vacc:VVI-25.12.SP2
# FROM harbor.vastaitech.com/ai_deliver/vllm_vacc:VVI-25.12.SP2_arm
# 设置工作目录
WORKDIR /test
# 设置环境变量
ENV XINFERENCE_SSE_PING_ATTEMPTS_SECONDS=864000
ENV VLLM_ENGINE_ITERATION_TIMEOUT_S=864000
ENV XOSCAR_CPU_AFFINITY=1
ENV XINFERENCE_RERANK_EMPTY_CACHE_COUNT=200
ENV XINFERENCE_EMBEDDING_EMPTY_CACHE_COUNT=200
ENV XINFERENCE_EMBEDDING_EMPTY_CACHE_TOKENS=81920
# 设置 pip 使用清华源（永久生效）
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN git clone https://github.com/xorbitsai/inference.git
WORKDIR /test/inference
RUN git checkout 00957020f29ee8ffd918eec877833d7904966ff0
RUN pip install -e .[vllm]
RUN pip install xoscar==0.8.1
RUN pip install qwen_omni_utils

# 可以添加其他需要的命令或配置
# ...

# 设置容器启动时的默认命令（根据你的需求修改）
CMD ["bash"]