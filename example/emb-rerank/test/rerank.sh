for i in {1..5}; do
  echo "Request $i:"
  curl -X 'POST' \
    'http://localhost:9999/v1/rerank' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "rerank_vacc",
      "query": "A man is eating pasta.",
      "documents": [
          "A man is eating food.",
          "A man is eating a piece of bread.",
          "The girl is carrying a baby.",
          "A man is riding a horse.",
          "A woman is playing violin."
      ],
      "return_documents": true
    }'
  echo -e "\n-------------------"
done
