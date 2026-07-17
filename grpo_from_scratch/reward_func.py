"""GRPO 示例使用的规则奖励函数。

每个函数都接收同一道题对应的若干 ``responses``，并为每条回答返回一个标量奖励。
训练脚本会把四种奖励相加：正确性提供强信号，数字与 XML 风格格式提供稠密的过程信号。
"""

import re


def extract_answer(text):
    """提取目标 ``<answer>...</answer>`` 区间中的最终答案文本。

    实现实际取最后一个 ``<answer>`` 之后、下一个 ``</answer>`` 之前的内容。例如：
    ``"<answer>\n 10 \n</answer>" -> "10"``。如果没有 ``<answer>``，则会把
    整段文本当成答案；这是宽松提取策略，格式正确与否由其它奖励单独判断。
    """
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def mark_num(text):
    """按关键标签的出现情况给部分格式分，最高为 0.5。

    四个标记各值 0.125。比如只正确生成 ``<think>\n`` 和 ``</think>\n``，
    即使还没有完整 answer 区块，也能得到 0.25，从而缓解完整格式奖励过于稀疏的问题。
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
    """最终答案与标准答案完全一致时奖励 2.0，否则为 0。

    这里是字符串精确匹配：标准答案为 ``"10"`` 时，``"10"`` 得分，
    但数学上等价的 ``"10.0"``、``"十"`` 或带单位的 ``"10只"`` 都不得分。
    """
    
    extracted_responses = [extract_answer(r) for r in responses]
    print(f"问题:\n{prompts[0]}", f"\n答案:\n{answers[0]}", f"\n模型输出:\n{responses[0]}", f"\n提取后的答案:\n{extracted_responses[0]}")
    return [2.0 if response == str(ans) else 0.0 for response, ans in zip(extracted_responses, answers)]

# 最终答案是否为纯数字的稠密奖励：即使算错，只要输出形态接近数学题答案也给部分分。
def digit_reward(prompts, responses, answers):
    """答案只含 Unicode 数字字符时奖励 0.5。

    例如 ``"10"`` 得 0.5；``"-2"``、``"3.14"``、``"10只"`` 均因包含
    非数字字符而得 0。这是刻意简化的教学规则，不是通用数值解析器。
    """
    extracted_responses = [extract_answer(r) for r in responses]
    return [0.5 if response.isdigit() else 0.0 for response in extracted_responses]

# 格式奖励
def hard_format_reward(prompts, responses, answers):
    """完整匹配指定的 think/answer 模板时奖励 0.5。

    正则带 ``^`` 和 ``$``，所以标签前后不能有额外文字；此外 ``.`` 默认不跨行，
    因而 think 与 answer 的正文各自只能占一行。例如下面的末尾换行也是必须的：
    ``<think>\n推导\n</think>\n<answer>\n10\n</answer>\n``。
    """
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    matches = [re.match(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]

# 标记奖励（改善格式奖励稀疏问题）
def mark_reward(prompts, responses, answers):
    """逐条调用 ``mark_num``，为部分正确的格式提供 0～0.5 的渐进奖励。"""
    return [mark_num(response) for response in responses]
