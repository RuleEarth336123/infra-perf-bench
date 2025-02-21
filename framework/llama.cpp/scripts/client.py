import openai

client = openai.OpenAI(
    base_url="http://localhost:12312/v1", # "http://<Your api-server IP>:port"
    api_key = "echo in the moon"
)

completion = client.chat.completions.create(
model="qwen2-0.5b",
messages=[
    {"role": "user", "content": "你好啊！怎么称呼你呢？"}
]
)

print(completion.choices[0].message)

# ./server \
#     -m /root/autodl-tmp/models/Llama3-8B-Chinese-Chat-GGUF/Llama3-8B-Chinese-Chat-q8_0-v2_1.gguf \
#     --host "127.0.0.1" \
#     --port 8080 \
#     -c 2048 \
#     -ngl 128 \
#     --api-key "echo in the moon"