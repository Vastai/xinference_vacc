for i in {1..5}; do
  echo "Request $i:"
  curl -X POST http://localhost:9998/v1/embeddings \
    -H "Content-Type: application/json" \
    -d '{
      "model": "emb_vacc",
      "input": "这是一个测试句子"
    }'
  echo -e "\n-------------------"
done
