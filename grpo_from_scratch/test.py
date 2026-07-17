"""调用训练后模型服务的最小冒烟测试。

运行前需要把 ``base_url``、``api_key`` 和 ``model`` 改成实际推理服务配置。
示例问题故意问“几只鸭”：题目只提到鸟和鸡，期望模型不要机械相加，而回答 0 只鸭。
"""

from openai import OpenAI

# 这里连接的是 OpenAI API 兼容服务，并不限定服务端必须使用 OpenAI 模型。
client = OpenAI(api_key='ww', base_url='http://10.250.2.24:8036/v1')

SYSTEM_PROMPT = """
按照如下格式回答问题：
<think>
你的思考过程
</think>
<answer>
你的回答
</answer>
"""

completion = client.chat.completions.create(
model = 'qwen1.5b',

# 评估时关闭随机性，便于重复检查同一 checkpoint 的输出。
temperature=0.0,
# 请求 token 对数概率；当前脚本只打印文本，若要分析置信度可继续读取返回的 logprobs。
logprobs = True,
messages=[
    {
        "role": "system", 
        "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": "天上五只鸟，地上五只鸡，一共几只鸭",
    }
],
)
print(completion.choices[0].message.content)
