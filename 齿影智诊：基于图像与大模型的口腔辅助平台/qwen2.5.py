from openai import OpenAI

client = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1/',
    api_key='d15696b8-7e0d-46bb-8492-5af7947b9bc0', # ModelScope Token
)

response = client.chat.completions.create(
    model='Qwen/Qwen2.5-VL-7B-Instruct', # ModelScope Model-Id
    messages=[{
        'role':
            'user',
        'content': [{
            'type': 'text',
            'text': '描述这幅图',
        }, {
            'type': 'image_url',
            'image_url': {
                'url':
                    'https://modelscope.oss-cn-beijing.aliyuncs.com/demo/images/audrey_hepburn.jpg',
            },
        }],
    }],
    stream=True
)

for chunk in response:
    print(chunk.choices[0].delta.content, end='', flush=True)