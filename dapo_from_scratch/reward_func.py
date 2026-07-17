"""教学版 DAPO/GRPO 奖励函数。

每个函数都接收等长的 ``prompts``、``responses``、``answers`` 列表，
并为组内每条回答返回一个标量奖励。``train.py`` 会把四项奖励相加。

当前最高总奖励为：

    correctness(2.0) + digit(0.5) + hard_format(0.5) + mark(0.5) = 3.5

这是一套便于小模型学习输出格式的 shaping reward，不是论文中只使用
正确/错误结果的原版 DAPO verifier。
"""

import re


def extract_answer(text):
    """提取最后一个 ``<answer>...</answer>`` 中的内容。

    例子：

        "<think>...</think><answer>42</answer>" -> "42"

    如果缺少 ``<answer>``，``split`` 会使整个文本被当作答案；因此正式
    实验通常还应显式检查标签是否存在。
    """
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def mark_num(text):
    """按照标签是否以指定换行形式出现，给予局部格式奖励。

    四个标记各值 0.125，最高 0.5。例如只正确生成 ``<think>\\n`` 和
    ``</think>\\n`` 时得到 0.25。这里使用 ``count(...) == 1``，所以重复
    标签不会获得该项奖励。
    """
    reward = 0
    if text.count("<think>\n") == 1:
        reward += 0.125
        
    if text.count("</think>\n") == 1:
        reward += 0.125
        
    if text.count("<answer>\n") == 1:
        reward += 0.125
        
    if text.count("</answer>\n") == 1:
        reward += 0.125
    return reward


# 生成答案是否正确的奖励
def correctness_reward(prompts, responses, answers):
    """最终答案与标准答案完全相等时给 2 分，否则给 0 分。

    例子：标准答案为整数 5 时，``"5"`` 得 2 分，``"05"`` 和 ``"5.0"``
    都得 0 分。这里没有进行数值等价或 LaTeX 归一化，规则比官方 DAPO
    verifier 更严格、更简单。
    """
    extracted_responses = [extract_answer(r) for r in responses]
    print(f"问题:\n{prompts[0]}", f"\n答案:\n{answers[0]}", f"\n模型输出:\n{responses[0]}", f"\n提取后的答案:\n{extracted_responses[0]}")
    return [2.0 if response == str(ans) else 0.0 for response, ans in zip(extracted_responses, answers)]


# 生成答案是否是数字的奖励（单纯依赖结果是否正确进行奖励，条件很苛刻，会导致奖励比较稀疏，模型难以收敛，所以加上答案是否是数字的奖励，虽然答案错误，但是至少生成的是数字（对于数学问题），也要给予适当奖励）
def digit_reward(prompts, responses, answers):
    """答案只包含十进制数字时给 0.5 分，用来缓解正确性奖励稀疏。

    ``"42"`` 可以得分；``"-1"``、``"3.14"``、``"1/2"`` 都不是
    ``str.isdigit()`` 意义上的纯数字，因此得 0 分。
    """
    extracted_responses = [extract_answer(r) for r in responses]
    return [0.5 if response.isdigit() else 0.0 for response in extracted_responses]


# 格式奖励
def hard_format_reward(prompts, responses, answers):
    """完整匹配 think/answer 模板时给 0.5 分。

    注意：正则中的 ``.`` 默认不跨越换行，并且模式要求结尾存在换行。
    因此多行思维链或没有尾部换行的回答可能得不到这项奖励。若要允许
    多行内容，可在确认训练目标后考虑 ``re.DOTALL``，此处保持原逻辑。
    """
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    matches = [re.match(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]


# 标记奖励（改善格式奖励稀疏问题）
def mark_reward(prompts, responses, answers):
    """调用 ``mark_num``，为部分正确的标签格式提供稠密奖励。"""
    return [mark_num(response) for response in responses]
