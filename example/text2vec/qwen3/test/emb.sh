for i in {1..5}; do
  echo "Request $i:"
  curl -X POST http://localhost:9996/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen3-Embedding-0.6B",
      "input": "这是一个测试句子"
    }'
  echo -e "\n-------------------"
done
