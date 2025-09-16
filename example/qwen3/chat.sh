curl -X POST "http://10.24.73.23:9998/v1/chat/completions" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer token-abc123" \
-d '{
  "model": "qwen3",
  "messages": [
    {"role": "system", "content": "你是一个专业助手"},
    {"role": "user", "content": "中国直辖市是哪里"}
  ],
  "stream": false
}'
