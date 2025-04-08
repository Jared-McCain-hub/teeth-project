from openai import OpenAI

client = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1/',
    api_key='d15696b8-7e0d-46bb-8492-5af7947b9bc0', # ModelScope Token
)

response = client.chat.completions.create(
    model='deepseek-ai/DeepSeek-V3-0324', # ModelScope Model-Id
    messages=[
        {
            'role': 'system',
            'content': 'You are a helpful assistant.'
        },
        {
            'role': 'user',
            'content': '你好'
        }
    ],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end='', flush=True)