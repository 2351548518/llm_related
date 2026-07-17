"""通过 OpenAI 兼容接口快速检查训练后模型的输出格式。

这个文件只做推理，不参与 DAPO 训练。运行前需要把 ``base_url``、
``api_key`` 和 ``model`` 改成实际服务配置。示例使用 ``temperature=0``
做确定性解码，主要检查模型能否按 think/answer 模板回答。
"""

from openai import OpenAI

# ``ww`` 是示例占位值；本地兼容服务可能不校验 key，远程服务则应使用
# 安全注入的真实凭据，不要把密钥硬编码到仓库。
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

# temperature=0 表示尽量使用确定性输出；训练阶段需要随机采样多个回答，
# 推理检查阶段则通常希望同一输入得到稳定结果。
temperature=0.0,
# 请求服务返回 token log-prob，便于进一步检查模型对输出 token 的置信度。
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

# 这里只打印最终文本；如果要分析 log-prob，可继续读取响应中的 logprobs 字段。
print(completion.choices[0].message.content)
