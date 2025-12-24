# curl -X POST "http://127.0.0.1:9998/v1/chat/completions" \
# -H "Content-Type: application/json" \
# -H "Authorization: Bearer token-abc123" \
# -d '{
#   "model": "qwen3",
#   "messages": [
#     {"role": "system", "content": "你是一个专业助手"},
#     {"role": "user", "content": "写一个1000字文章"}
#   ],
#   "stream": true
# }'
#!/bin/bash

# 无限循环请求
while true; do
    curl -X POST "http://192.168.28.113:9996/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer token-abc123" \
    -d '{
      "model": "qwen3",
      "messages": [
        {"role": "system", "content": "你是一个专业助手"},
        {"role": "user", "content": "写一个1000字文章"}
      ],
      "stream": true
    }'
    
    echo ""  # 换行分隔
done