from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ============================================================
# s1 风格的 test-time scaling 演示：用 "Wait" 注入延长思考、自我纠错
# 论文：s1: Simple Test-Time Scaling（budget forcing 技巧）
# ============================================================

tokenizer = AutoTokenizer.from_pretrained("/home/user/Downloads/DeepSeek-R1-Distill-Qwen-1.5B")
llm = LLM(model="/home/user/Downloads/DeepSeek-R1-Distill-Qwen-1.5B", gpu_memory_utilization=0.15)

# ---------- 第一套采样参数：无停止符，让模型自由生成到 max_tokens ----------
# 用来看模型"原始的、未受干预的"一次完整输出
sampling_params = SamplingParams(
    temperature=0,                # 贪心解码，保证可复现
    max_tokens=32768,
    skip_special_tokens=False    # 保留 <|im_start|>/<|im_end|>/<think> 等特殊 token，便于拼接
)

# 测试题：9.11 和 9.8 谁大？（经典容易答错的题，用来检验纠错效果）
prompt = '9.11和9.8谁大？'
# 套上 Qwen 的 chat template，构造一个完整的对话 prompt
prompt = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"

# 模型原始输出部分
outputs = llm.generate(
    prompt,
    sampling_params
)
print(f'原始输出：{prompt}{outputs[0].outputs[0].text}')
print('+'*20)

# ---------- 第二套采样参数：遇到 </think> 就停 ----------
# 关键点：只取"思考过程"这一段（到 </think> 为止），
# 这样后续追加 "Wait" 时，它落在思考过程内部，能诱导模型继续思考而不是直接给出答案。
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=32768,
    stop='</think>',               # 在思考结束标签处截断
    skip_special_tokens=False
)

# 用新的停止策略再生成一次，截取到 </think> 之前的思考内容
outputs = llm.generate(
        prompt,
        sampling_params
    )

# ---------- "Wait" 注入（budget forcing）核心步骤 ----------
# 在模型自己的思考输出后面强行拼一个 "Wait"，
# 模型读到 "Wait" 会本能地"等等，我再想想"，从而延长思考链、修正前面的错误。
wait = 'Wait'
for i in range(1):                 # 这里只注入 1 次；调大可做多轮"反思-修正"，思考预算越高答案通常越稳
    prompt += outputs[0].outputs[0].text + wait   # 把上一轮思考 + "Wait" 拼回 prompt

    outputs = llm.generate(        # 让模型在 "Wait" 之后续写思考
        prompt,
        sampling_params            # 仍以 </think> 为停止符
    )

print(f'wait后的输出：{prompt}{outputs[0].outputs[0].text}')
print('+'*20)
# 把"Wait 续写"得到的思考也并入 prompt，作为最终生成的基础上下文
prompt += outputs[0].outputs[0].text

# ---------- 第三套采样参数：以 <|im_end|> 为真正结束标志，生成最终答案 ----------
stop_token_ids = tokenizer("<|im_end|>")["input_ids"]   # 取 <|im_end|> 的 token id 列表
sampling_params = SamplingParams(
    max_tokens=32768,
    min_tokens=0,                 # 允许模型立刻停（不强制最小长度）
    stop_token_ids=stop_token_ids,# 遇到 <|im_end|> 才正式结束整段对话
    skip_special_tokens=False,
    temperature=0.0,
)
outputs = llm.generate(
    prompt,
    sampling_params=sampling_params,
)

print(f'最后的输出：{prompt}{outputs[0].outputs[0].text}')
