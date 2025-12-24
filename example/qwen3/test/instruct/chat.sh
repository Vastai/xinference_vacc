curl -X POST "http://127.0.0.1:9999/v1/chat/completions" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer token-abc123" \
-d '{
  "model": "Qwen3-Instruct",
  "messages": [
    {"role": "system", "content": "你是一个专业助手"},
    {"role": "user", "content": "中国直辖市是哪里"}
  ],
  "stream": false
}'
