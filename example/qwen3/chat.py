from openai import OpenAI
client = OpenAI(base_url="http://localhost:9997/v1", api_key="EMPTY")

response = client.chat.completions.create(
  model="qwen3",
  messages=[{"role": "user", "content": "中国直辖市是哪里"}],
  temperature=0.5,
)
print(response.choices[0].message.content)