"""使用 TRL 的 GRPOTrainer 做数学推理强化学习的教学示例。

这份脚本只演示 DeepSeek-R1/R1-Zero 中“按规则奖励进行推理强化学习”的核心思路，
并不是完整的 DeepSeek-R1 训练复现：它没有实现冷启动 SFT、拒绝采样、第二阶段 SFT
和通用对齐 RL，而且实际起点是 Qwen2.5-Instruct，而不是未经指令微调的基础模型。

主流程：
1. 把 GSM8K 中文题目转换成聊天 prompt；
2. 对每个 prompt 采样多条回答；
3. 用格式、数字类型和答案正确性等规则打分；
4. GRPO 根据同组回答之间的相对奖励更新模型。
"""

import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import trl
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, TaskType

# 说明：torch、Dataset 和 trl 模块名在当前主流程中没有被直接引用；PEFT 的三个名称
# 只有启用下面的 LoRA 代码时才会用到。这里保留原导入，避免改变原脚本结构。

# =============================================================================
# 该脚本借鉴 DeepSeek-R1-Zero 的核心思想：用 GRPO 和规则奖励强化推理能力。
# 论文路线：pretrain -> rl（R1-Zero）
# 本实现：Qwen2.5-0.5B-Instruct + GSM8K中文 + 规则奖励(GRPO)
# 注意：Qwen2.5-0.5B-Instruct 已经过指令微调，所以这里不是严格意义上的“跳过 SFT”。
#
# GRPO（Group Relative Policy Optimization）要点：
#   - 对同一个 prompt 采样 G 条回答（这里 num_generations=16）
#   - 用奖励函数给每条打分，组内做归一化得到优势 A = (r - mean) / std
#   - 用优势更新策略，免去传统 PPO 中额外的 value/critic 网络
#   - 论文中 R1 的 RL 一阶段用"基于规则的奖励"，这里用 5 个规则奖励函数实现
# =============================================================================

# 系统提示词：强制模型输出 <think>...</think><answer>...</answer> 的格式
# 这是 R1 "格式规范" 目标的简化版，让推理过程可控、可解析。
SYSTEM_PROMPT = """
按照如下格式生成：
<think>
...
</think>
<answer>
...
</answer>
"""
# 示例：模型对 "2+3=?" 的期望输出大致如下：
#   <think>
#   2加3等于5
#   </think>
#   <answer>
#   5
#   </answer>


def process_data(data):
    """把原始 GSM8K 数据改造成 GRPO 需要的格式。

    GRPOTrainer 期望每条样本含两个字段：
      - prompt:  chat 消息列表 [system, user, ...]
      - answer:  标准答案（字符串），供奖励函数比对，不会喂给模型

    输入示例：
      {"question_zh-cn": "小明有 2 个苹果，又买了 3 个，一共有几个？",
       "answer_only": "5"}

    新增字段示例：
      prompt = [
          {"role": "system", "content": SYSTEM_PROMPT},
          {"role": "user", "content": "小明有 2 个苹果，又买了 3 个，一共有几个？"}
      ]
      answer = "5"

    Dataset.map 默认保留原始列，因此 question_zh-cn、answer_only 等字段仍可能存在；
    GRPOTrainer 只会把它需要的字段以及奖励函数同名参数取出来使用。
    """
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question_zh-cn']}
        ],
        'answer': x['answer_only']
    })
    return data


def extract_answer(text):
    """从模型输出中抽取 <answer>...</answer> 之间的内容。

    例：text = "<think>...</think>\n<answer>\n5\n</answer>"
        先 split("<answer>") 取最后一段 -> "\n5\n</answer>"
        再 split("</answer>")[0]  -> "\n5\n"
        strip()                  -> "5"

    边界行为：若输出里没有 <answer>，第一次 split 会返回整段文本；若没有
    </answer>，则会取 <answer> 之后的全部剩余文本。因此它是宽容解析，不负责验证格式。
    """
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def mark_num(text):
    """标记奖励：每出现一个"正确闭合的标签"就给一点分，缓解奖励稀疏。

    论文中 RL 初期模型几乎无法产出完全正确的格式，hard_format 一刀切 0/0.5 会导致
    大部分样本拿不到信号、难以收敛。这里把格式拆成 4 个小目标，每命中一个 +0.125，
    上限 0.5，给模型一个"逐步逼近正确格式"的稠密学习信号。

    例：
      "<think>\n3+2=5\n</think>\n<answer>\n5\n</answer>\n"  -> 4 个标签全中 -> 0.5
      "<think>\n3+2=5\n</think>\n<answer>5</answer>"        -> 后两个未命中 -> 0.25
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
def correctness_reward(prompts, completions, answer, **kwargs):
    """正确性奖励（权重最大 2.0）：提取模型答案与标准答案做精确字符串匹配。

    GRPOTrainer 会把同一 batch 内所有 completion 传进来，这里按样本逐条比较。
    answer 会用 str(ans) 转成字符串；模型答案经过 strip，所以首尾空白不会影响比较。
    但 "5.0"、"答案是5" 与标准答案 "5" 仍不相等，这是一种较严格、较稀疏的奖励。

    例：answer[0]="5"，模型输出提取出 "5" -> 奖励 2.0；提取出 "6" -> 0.0

    completions 的形状示例（每条 completion 是一段 assistant 消息列表）：
      [[{"role": "assistant", "content": "<think>...</think><answer>5</answer>"}], ...]
    所以 completion[0]['content'] 才是实际生成文本。
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    print(f"问题:\n{prompts[0][-1]['content']}", f"\n答案:\n{answer[0]}", f"\n模型输出:\n{responses[0]}", f"\n提取后的答案:\n{extracted_responses[0]}")
    return [2.0 if response == str(ans) else 0.0 for response, ans in zip(extracted_responses, answer)]


# 生成答案是否是数字的奖励（单纯依赖结果是否正确进行奖励，条件很苛刻，会导致奖励比较稀疏，模型难以收敛，所以加上答案是否是数字的奖励，虽然答案错误，但是至少生成的是数字（对于数学问题），也要给予适当奖励）
def digit_reward(completions, **kwargs):
    """数字奖励（0.5）：只要提取出的答案是纯数字就给分。

    动机：数学题里 correctness_reward 只在"答案完全正确"时给 2.0，初期模型几乎全错，
    组内 16 条可能全是 0，优势归一化后区分度为 0，学不动。引入数字奖励后，
    "答错但至少生成了数字"也能拿到 0.5，提供中间信号引导模型先学会"输出一个数字"。

    例：提取 "5"  -> 0.5；提取 "abc" -> 0.0；提取 "3.14" -> 0.0（isdigit 对小数点返回 False）
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    return [0.5 if response.isdigit() else 0.0 for response in extracted_responses]


# 格式奖励
def hard_format_reward(completions, **kwargs):
    """硬格式奖励（0.5）：用严格正则从头到尾完整匹配。

    ^<think>\n 开头 ... \n</think>\n<answer>\n ... \n</answer>\n$ 结尾，中间用 .*? 非贪婪。
    要求换行位置完全一致。命中即 0.5，否则 0。信号最稀疏，作为"最终目标"。

    例：
      "<think>\n3+2=5\n</think>\n<answer>\n5\n</answer>\n" -> match -> 0.5
      "<think>3+2=5</think><answer>5</answer>"            -> 不匹配 -> 0.0

    重要限制：这里没有传 re.DOTALL，正则中的点号不能跨越换行。因此 think 或 answer
    内部若有多行内容，通常会匹配失败；当前规则实际上更适合“单行推理 + 单行答案”。
    """
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]


# 格式奖励
def soft_format_reward(completions, **kwargs):
    """软格式奖励（0.5）：正则更宽松，只要求标签先后顺序对，不强求换行。

    <think>.*?</think>\\s*<answer>.*?</answer>，\\s* 容许标签间任意空白。
    命中即 0.5。比 hard 更容易拿分，作为中间过渡信号。

    例：
      "<think>xxx</think><answer>5</answer>"       -> 0.5
      "<think>xxx</think>  <answer>5</answer>"     -> 0.5
      "<answer>5</answer><think>...</think>"       -> 0.0（顺序错）

    注意：虽然名为“软格式”，它同样没有启用 re.DOTALL，所以上面单行示例能命中，
    SYSTEM_PROMPT 所示的多行输出反而可能得 0 分。re.match 还要求匹配从字符串开头开始。
    """
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]


# 标记奖励（改善格式奖励稀疏问题）
def mark_reward(completions, **kwargs):
    """标记奖励（0.5）：见 mark_num，逐标签给分。"""
    responses = [completion[0]['content'] for completion in completions]
    return [mark_num(response) for response in responses]


if __name__ == '__main__':
    # 下面两个路径都来自原作者的 Linux 环境；在其他机器运行时需要替换成本地路径。
    model_name = "/home/user/Downloads/Qwen2.5-0.5B-Instruct"

    # 默认执行全参数训练。from_pretrained 此处没有显式指定 torch_dtype，随后再整体搬到 GPU。
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # 如果使用lora方法训练，取消如下注释
    # lora_config = LoraConfig(
    # r=8,              # 低秩矩阵的秩；越大表达能力越强，但参数量和显存占用也越高
    # lora_alpha=256,   # LoRA 缩放系数；常见缩放比例与 lora_alpha / r 有关
    # # 同时给注意力投影层和 MLP 投影层注入 LoRA 适配器
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # lora_dropout=0.1, # 只作用于 LoRA 分支的 dropout，用于缓解过拟合
    # task_type=TaskType.CAUSAL_LM)
    # # 包装后通常只有 LoRA 适配器参数参与训练，显存需求显著低于全参数训练
    # model = get_peft_model(model, lora_config)
    # 假定存在 CUDA GPU；若硬件不支持 CUDA 或 BF16，当前配置不能直接运行。
    model.cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 加载 GSM8K 中文版（小学应用题），answer_only 字段是纯数字答案，便于 correctness_reward 比对
    ds = load_dataset('/home/user/wyf/deepseek_learn/gsm8k_chinese')
    data = process_data(ds['train'])

    output_dir="output"

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=5e-6,          # 小学习率，RL 阶段避免破坏预训练能力
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,          # 前 10% 步数线性升温，稳定初期
        lr_scheduler_type='cosine',  # 余弦退火
        logging_steps=1,
        bf16=True,                   # 混合精度节省显存
        per_device_train_batch_size=1,
        # 单卡时每 4 个微批次更新一次；多卡时全局有效 batch 还要乘进程/GPU 数量。
        gradient_accumulation_steps=4,
        num_generations=16,          # GRPO 关键：每个 prompt 采样 16 条做组内比较
        temperature=1.0,             # rollout 采样温度：保持组内多样性，为相对优势提供区分信号
        max_prompt_length=256,       # prompt 截断长度
        max_completion_length=200,    # 生成截断长度，控制 think+answer 总长
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,           # 梯度裁剪，RL 训练稳定性关键
        log_on_each_node=False,
        use_vllm=False,              # 是否用 vLLM 加速生成
        report_to="tensorboard"      # 训练日志上报 TB，可视化 reward 曲线
    )

    # GRPOTrainer 会按列表顺序调用 5 个奖励函数，并将结果相加（未设置 reward_weights）。
    # 从各函数的标称上限相加看是 4.0；但按当前正则，hard_format 所要求的多行格式
    # 无法同时通过未启用 DOTALL 的 soft_format，所以实际可达上限低于 4.0。
    #
    # 以“正确、纯数字、严格单行内容格式”的回答为例：
    #   mark=0.5 + soft=0.0 + hard=0.5 + digit=0.5 + correctness=2.0 = 3.5
    # 这也说明辅助奖励可能压过任务目标：即使答案错误，格式和数字奖励仍可拿到约 1.5。
    # GRPO 在组内对总奖励归一化得优势，再更新策略
    trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        mark_reward,
        soft_format_reward,
        hard_format_reward,
        digit_reward,
        correctness_reward
        ],
    args=training_args,
    train_dataset=data,

)
    trainer.train()
    trainer.save_model(output_dir)
