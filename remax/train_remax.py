"""使用 ReMax baseline、KL 约束和 PPO 裁剪目标训练语言模型。

阅读这份脚本时，可以把一次参数更新理解成下面 5 步：

1. policy 对同一批问题生成两份答案：一份随机采样，一份贪心生成。
2. reward_model 分别给两份答案打分，二者之差作为 ReMax 的相对奖励。
3. ref_policy 计算参考概率，用来惩罚 policy 偏离初始模型太远。
4. 将相对奖励和逐 token KL 惩罚组合成 advantage/return。
5. 使用 PPO clipped loss 更新 policy；ref_policy 始终不参与优化。

例如，随机答案答对得到 +1，贪心答案答错得到 -1，则 ReMax 相对奖励为
``+1 - (-1) = +2``。如果二者都答对或都答错，相对奖励就是 0。

本文件只定义训练逻辑。它期望数据集中的每条样本形如：

    {
        "prompt": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "Natalia 一共卖出多少个夹子？"},
        ],
        "answer": "72",
    }

注意：注释中的 shape 默认省略词表维度，并以单进程训练为例；多卡训练时
Accelerate 会把全局 batch 分发到各个进程。
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import time
from peft import get_peft_model, LoraConfig, TaskType
import pandas as pd
import datasets
import copy
import torch
from typing import List, Dict, Any
import math

import gc
import math
import os
import re
import textwrap
import time
from collections import defaultdict
from typing import Callable, Optional, Union
from dataclasses import field, dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import broadcast, gather_object
from datasets import Dataset
from torch.utils.data import DataLoader

from transformers import (
    BaseImageProcessor,
    DataCollatorWithPadding,
    FeatureExtractionMixin,
    GenerationConfig,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
    TrainerControl,
    is_wandb_available,
)

from transformers.integrations import get_reporting_integration_callbacks
from transformers.trainer import DEFAULT_CALLBACKS, DEFAULT_PROGRESS_CALLBACK
from transformers.trainer_callback import CallbackHandler, ExportableState, PrinterCallback

from trl.trainer.utils import (OnlineTrainerState,OnPolicyConfig,
    batch_generation,
    disable_dropout_in_model,
    exact_div,
    first_true_indices,
    forward,
    get_reward,
    prepare_deepspeed,
    print_rich_table,
    selective_log_softmax,
    truncate_response,
    log_table_to_comet_experiment)

from trl.models.utils import unwrap_model_for_generation
# 填充位置没有真实 token 概率，用固定值占位；后续 loss 依靠 padding_mask 忽略它们。
INVALID_LOGPROB = 1.0

# LoRA 的低秩维度。rank 越大，通常表达能力和显存占用都越高。
# 例如，rank=32 比 rank=8 拥有更多可训练参数，但仍远少于全量微调。
lora_rank = 32


# ============================= 奖励函数 =============================
def extract_answer(text):
    """提取模型输出中第一个 ``<answer>...</answer>`` 片段的正文。

    示例：
        输入：``"<think>48 + 24</think><answer>72</answer>"``
        输出：``"72"``

    若缺少 ``<answer>``，当前实现会把整段文本视作答案；若缺少结束标签，
    会取 ``<answer>`` 后的全部内容。这是 ``split`` 写法本身的行为。
    """
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def correctness_reward(prompts=None, completions=None, answer=None, **kwargs):
    """按最终答案是否精确相等返回 +1/-1 奖励。

    参数由 Trainer 按 batch 传入。例如：

        completions = ["<answer>72</answer>", "<answer>9</answer>"]
        answer = ["72", "10"]
        返回值 = [1, -1]

    这里是字符串精确比较，空格、逗号、小数格式和数据类型都会影响结果。
    例如字符串 ``"72"`` 与整数 ``72`` 并不相等，准备数据时应统一类型。
    """
    responses = [extract_answer(completion) for completion in completions]
    # 仅用于观察采样结果；多卡训练时每个进程都可能打印一份。
    print(f"模型输出：{completions[0]}")
    return [1 if response==ans else -1 for response, ans in zip(responses, answer)]


class DataCollator:
    """把若干数据样本整理成 Trainer 使用的文本 batch。

    与监督微调中的 collator 不同，这里暂时不做 tokenize，因为生成前还需要
    对整批聊天消息调用 ``tokenizer.apply_chat_template``。

    示例输入：
        [{"prompt": chat_a, "answer": "72"},
         {"prompt": chat_b, "answer": "10"}]

    示例输出：
        {"prompt": [chat_a, chat_b], "answer": ["72", "10"]}
    """
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 保持 prompt 为聊天消息列表，保持 answer 为原始标准答案。
        batch = {"prompt":[feature['prompt'] for feature in features], "answer":[feature['answer'] for feature in features]}
        return batch



@dataclass
class ReMaxConfig(OnPolicyConfig):
    """ReMaxTrainer 的超参数。

    继承自 TRL 的 ``OnPolicyConfig``，因此 batch size、生成长度、温度等通用
    参数来自父类；这里补充 PPO、KL、奖励裁剪和 ReMax 回报所需的配置。
    """

    exp_name: str = field(
        default=os.path.basename(__file__)[:-3],
        metadata={"help": "Name of this experiment."},
    )
    num_ppo_epochs: int = field(
        default=1,
        metadata={"help": "Number of epochs to train."},
    ) # 一个batch 的数据 优化 几次模型
    kl_coef: float = field(
        default=0.05,
        metadata={"help": "KL coefficient."},
    )
    cliprange: float = field(
        default=0.2,
        metadata={"help": "Clip range."},
    )
    reward_clip_range: float = field(
        default=10.0,
        metadata={"help": "Clip range for rewards"},
    )
    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation."
        },
    )
    gamma: float = field(
        default=0.95,
        metadata={"help": "Discount factor."},
    ) # 折扣 因子 cumulative_reward *= self.args.gamma



class ReMaxTrainer(Trainer):
    """负责采样经验、计算 ReMax 回报并更新 policy 的在线 Trainer。

    主要对象：

    - ``policy``：需要训练的模型，主程序中是加了 LoRA 的 Qwen。
    - ``ref_policy``：固定参考模型，只用于计算 KL，不执行 optimizer.step()。
    - ``reward_model``：可为神经网络，也可为奖励函数列表。
    - ``processing_class``：通常是 tokenizer，负责聊天模板、编码和解码。

    一个 rollout batch 会先完整生成并打分，再被切成 mini/micro batch 更新。
    这样生成阶段和优化阶段使用的是同一批固定经验。
    """

    
    def __init__(
        self,
        config: ReMaxConfig,
        processing_class: Optional[
            Union[PreTrainedTokenizerBase, BaseImageProcessor, FeatureExtractionMixin, ProcessorMixin]
        ],
        policy: nn.Module,
        ref_policy: nn.Module,
        reward_model: Union[nn.Module, Callable[[list[str]], list[float]]],
        train_dataset: Dataset,
        data_collator: Optional[DataCollatorWithPadding] = None,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        # less commonly used
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        callbacks: Optional[list[TrainerCallback]] = None,
    ) -> None:
        """初始化模型、批大小、优化器、回调以及训练/验证 DataLoader。"""
        if ref_policy is policy:
            raise ValueError(
                "`policy` and `ref_policy` cannot be the same object. If you want `ref_policy` to be the "
                "same as `policy`, you must mass a copy of it, or `None` if you use peft."
            )

        self.args = config
        args = config
        self.processing_class = processing_class
        self.policy = policy


        if data_collator is None:
            data_collator = DataCollatorWithPadding(self.processing_class)

       
        self.ref_policy = ref_policy
        self.reward_model = reward_model
        self.train_dataset = train_dataset
        self.train_dataset_len = len(train_dataset)
        self.data_collator = data_collator
        self.eval_dataset = eval_dataset
        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = None  # needed for transformers >= 4.47

        #########
        # 计算多卡训练中的各种 batch size
        #########
        if args.total_episodes is None:  # allow the users to define episodes in terms of epochs.
            args.total_episodes = int(args.num_train_epochs * self.train_dataset_len)
        accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
        self.accelerator = accelerator
        args.world_size = accelerator.num_processes
        # local_batch_size：单个进程完成一次 rollout/update 所需的样本数。
        # 例：per_device=1、gradient_accumulation=8、num_mini_batches=1，结果为 8。
        args.local_batch_size = (
            args.per_device_train_batch_size * args.gradient_accumulation_steps * args.num_mini_batches
        )
        # micro_batch_size 是所有进程在一次前向中共同处理的样本数。
        args.micro_batch_size = int(args.per_device_train_batch_size * args.world_size)
        # batch_size 是一次 update 在所有进程上总共消耗的 rollout 样本数。
        args.batch_size = int(args.local_batch_size * args.world_size)
        # mini batch 用于 PPO 切分；每个 mini batch 还会再切成 micro batch 做梯度累积。
        args.mini_batch_size = exact_div(
            args.batch_size, args.num_mini_batches, "`batch_size` must be a multiple of `num_mini_batches`"
        )
        args.local_mini_batch_size = exact_div(
            args.local_batch_size, args.num_mini_batches, "`local_batch_size` must be a multiple of `num_mini_batches`"
        )
        args.num_total_batches = math.ceil(
            args.total_episodes / args.batch_size
        )  # we may train for more than `total_episodes`
        time_tensor = torch.tensor(int(time.time()), device=accelerator.device)
        time_int = broadcast(time_tensor, 0).item()  # avoid different timestamps across processes
        args.run_name = f"{args.exp_name}__{args.seed}__{time_int}"
        self.local_seed = args.seed + accelerator.process_index * 100003  # Prime
        if args.num_sample_generations > 0:
            self.sample_generations_freq = max(1, args.num_total_batches // args.num_sample_generations)
        

        #########
        # 配置模型、优化器和生成停止条件
        #########
        # 在线 RL 训练中关闭 dropout，可让 rollout 概率与随后重算的概率更稳定。
        for module in [policy, ref_policy, reward_model]:
            if isinstance(module, nn.Module):
                disable_dropout_in_model(module)
        if args.stop_token and args.stop_token == "eos":
            args.stop_token_id = self.processing_class.eos_token_id
        self.model = policy
        self.create_optimizer_and_scheduler(
            num_training_steps=args.num_total_batches
        )  # note that we are calling `self.lr_scheduler.step()` manually only at the batch level

        #########
        ### 初始化 Hugging Face Trainer 的日志、回调和状态对象
        #########
        default_callbacks = DEFAULT_CALLBACKS + get_reporting_integration_callbacks(self.args.report_to)
        self.callbacks = default_callbacks if callbacks is None else default_callbacks + callbacks
        self.callback_handler = CallbackHandler(
            self.callbacks, self.model, self.processing_class, self.optimizer, self.lr_scheduler
        )
        self.add_callback(PrinterCallback if self.args.disable_tqdm else DEFAULT_PROGRESS_CALLBACK)
        self.control = TrainerControl()
        self.state = OnlineTrainerState(
            is_local_process_zero=self.is_local_process_zero(),
            is_world_process_zero=self.is_world_process_zero(),
            stateful_callbacks=[
                cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
            ],
        )

        self.current_flos = 0
        self.hp_search_backend = None
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None
        # Create distant repo and output directory if needed
        self.hub_model_id = None
        if self.args.push_to_hub:
            self.init_hf_repo()
        if self.args.should_save:
            os.makedirs(self.args.output_dir, exist_ok=True)
        self.backup_model = None


        #########
        ### 创建 DataLoader
        #########
        # drop_last=True 保证每个 rollout batch 大小固定，否则最后一个小 batch
        # 无法按照 local_mini_batch_size 和 per_device_train_batch_size 等分。
        self.dataloader = DataLoader(
            self.train_dataset,
            batch_size=args.local_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=True,  # needed; otherwise the last batch will be of ragged shape
        )
        # sync random states for DataLoader(shuffle=True) before `accelerator.prepare`
        # see https://gist.github.com/vwxyzjn/2581bff1e48e185e0b85b6dfe1def79c
        torch.manual_seed(args.seed)
        self.model, self.optimizer, self.dataloader = accelerator.prepare(self.model, self.optimizer, self.dataloader)
        torch.manual_seed(self.local_seed)  # reset the local seed again

        # 验证集不参与反向传播，只用于 generate_completions 展示回答和奖励。
        # 当前实现会无条件创建该 DataLoader，因此调用方应提供非空 eval_dataset。
        self.eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=True,
        )  # no need to shuffle eval dataset
        self.eval_dataloader = accelerator.prepare(self.eval_dataloader)

        if self.is_deepspeed_enabled:
            if isinstance(self.reward_model, nn.Module):
                self.reward_model = prepare_deepspeed(
                    self.reward_model, args.per_device_train_batch_size, args.fp16, args.bf16
                )
            self.ref_policy = prepare_deepspeed(
                self.ref_policy, args.per_device_train_batch_size, args.fp16, args.bf16
            )
            self.deepspeed = self.model
        else:
            self.ref_policy = self.ref_policy.to(self.accelerator.device)
            if isinstance(self.reward_model, nn.Module):
                self.reward_model = self.reward_model.to(self.accelerator.device)

    def get_train_dataloader(self) -> DataLoader:
        return self.dataloader

    def get_eval_dataloader(self) -> DataLoader:
        return self.eval_dataloader
    
    def train(self):
        """执行完整的在线训练循环。

        核心张量的典型 shape：

        - ``queries``：``[batch, prompt_length]``
        - ``responses``：``[batch, response_length]``
        - ``logprobs``：``[batch, response_length]``，只保存实际采样 token 的概率
        - ``scores``：``[batch]``，随机回答的序列级奖励
        - ``returns``：``[batch, response_length]``，分配到每个生成 token 的训练信号

        整个 rollout 位于 ``torch.no_grad()`` 中，不在生成时保存计算图；真正的
        带梯度前向发生在 PPO micro batch 更新阶段。
        """
        args = self.args
        accelerator = self.accelerator
        optimizer = self.optimizer
        model = self.model
        self.model_wrapped = self.model
        ref_policy = self.ref_policy
        reward_model = self.reward_model
        processing_class = self.processing_class
        dataloader = self.dataloader
        device = accelerator.device

        def repeat_generator():
            # 当 total_episodes 超过数据集长度时，从打乱后的 DataLoader 继续循环取数。
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())

        # 随机生成配置：这是被训练的行为样本。temperature 越高，候选答案越多样。
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True
        )
        
        # 贪心生成配置：每一步都选概率最高的 token，用作 ReMax baseline。
        # 同一个问题同时生成随机答案和贪心答案，能减少问题难度带来的奖励方差。
        greedy_generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            do_sample=False
        )
        

        accelerator.print("===training policy===")
        start_time = time.time()
        
        # 训练过程记录的变量。三个维度依次对应 PPO epoch、mini batch、
        # gradient accumulation step，最后在一次 update 结束时取均值写日志。
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)
        
        model.train()

        # 初始化 Trainer 状态；episode 统计已消费的样本数，global_step 统计 update 数。
        self.state.global_step = 0
        self.state.episode = 0
        self.state.max_steps = (args.num_total_batches * args.num_mini_batches) // 2
        self.state.num_train_epochs = args.total_episodes / self.train_dataset_len
        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(self.state.max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(self.state.max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(self.state.max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps
        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader)
            with torch.no_grad():
                queries = data["prompt"]
                answers = data["answer"]
                # decoder-only 模型批量生成时应左侧 padding，保证每条 prompt 的最后
                # 一个有效 token 都紧邻首个生成 token。
                processing_class.padding_side = "left"
                # add_generation_prompt=True 会添加 Qwen assistant 起始标记。
                # 例：两条聊天消息 -> queries.shape 可能为 [8, 96]。
                queries = processing_class.apply_chat_template(queries, tokenize=True, add_generation_prompt=True, return_tensors='pt', padding=True)
                queries = queries.to(device)

                # query_responses 会包含“原 prompt + 新生成 token”；记录 prompt 长度后，
                # 可以通过 [:, context_length:] 精确切出 response。
                context_length = queries.shape[1]
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                baseline_scores = []
                sequence_lengths = []

                # 一次生成随机回答和贪心 baseline。unwrap 后可兼容 Accelerate、
                # DeepSpeed 等包装器，并在需要时临时收集 ZeRO-3 参数。
                with unwrap_model_for_generation(
                    self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model:
                    # 生成 响应
                    query_responses, logitss = batch_generation(
                        unwrapped_model,
                        queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                    )
                    
                    # 生成 贪心 baseline 的 响应
                    greedy_query_responses, greedy_logitss = batch_generation(
                        unwrapped_model,
                        queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        greedy_generation_config,
                    )

                # 将大 batch 切成 rollout forward 小 batch，降低生成、参考模型前向
                # 和奖励模型前向的峰值显存。最终会在下面重新 cat 回完整 batch。
                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                  
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    answer = answers[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]
                    greedy_query_response = greedy_query_responses[i : i + args.local_rollout_forward_batch_size]
                    response = query_response[:, context_length:]
                    greedy_response = greedy_query_response[:, context_length:]


                    # selective_log_softmax 不保留整个词表的概率，只取“实际采到的 token”
                    # 的 log probability，结果 shape 与 response 相同。
                    logits = logitss[i : i + args.local_rollout_forward_batch_size]
                    logprob = selective_log_softmax(logits, response)
                    del logits
                    torch.cuda.empty_cache()

                    # ref_policy 只评估随机回答，不生成新答案。切片 [context_length-1:-1]
                    # 将每个生成 token 与“预测它的前一个位置 logits”对齐。
                    ref_output = forward(ref_policy, query_response, processing_class.pad_token_id)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_logprob = selective_log_softmax(ref_logits, response)
                    del ref_output, ref_logits
                    torch.cuda.empty_cache()

                    # 回答处理 1：遇到首个 EOS/stop token 后，把剩余位置改为 padding。
                    # 例：[思考, 72, EOS, 噪声] -> [思考, 72, EOS, PAD]。
                    postprocessed_response = response
       
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )
                        
                        postprocessed_greedy_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, greedy_response
                        )
                    
                    # 回答处理 2：分别给随机回答和贪心回答打分。
                    # sequence_length 是最后一个非 PAD token 的下标，用于构建 mask/return。
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    postprocessed_greedy_query_response = torch.cat((query, postprocessed_greedy_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1

                    if isinstance(reward_model, nn.Module):
                        _, score, _ = get_reward(
                            reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                        )
                        _, baseline_score, _ = get_reward(
                            reward_model, postprocessed_greedy_query_response, processing_class.pad_token_id, context_length
                        )
                   
                    elif isinstance(reward_model, list):
                        # 支持组合多个奖励：每列对应一个 reward model/function，最后求和。
                        # 例：[格式奖励 0.5, 正确性奖励 1.0] -> 总奖励 1.5。
                        scores_ = torch.zeros((query.shape[0], len(reward_model)))
                        baseline_scores_ = torch.zeros((query.shape[0], len(reward_model)))
                        
                        for i, rm in enumerate(reward_model):
                            if isinstance(rm, nn.Module): # 基于 模型 的 奖励
                                _, score, _ = get_reward(rm, postprocessed_query_response, processing_class.pad_token_id, context_length)
                                _, baseline_score, _ = get_reward(rm, postprocessed_greedy_query_response, processing_class.pad_token_id, context_length)
                                
                                scores_[:, i] = score
                                baseline_scores_[:, i] = baseline_score
                            
                            else: # 基于 规则 的 奖励
                                # Python 奖励函数接收解码后的字符串；本脚本使用
                                # correctness_reward 检查 <answer> 中的最终答案。

                                # 转换成 文本
                                response_text = processing_class.batch_decode(postprocessed_response, skip_special_tokens=True)
                                greedy_response_text = processing_class.batch_decode(postprocessed_greedy_response, skip_special_tokens=True)
                                
                                # 计算奖励
                                scores_[:, i] = torch.tensor(rm(completions=response_text, answer=answer))
                                baseline_scores_[:, i] = torch.tensor(rm(completions=greedy_response_text, answer=answer))
                                
                        # 当前 句子 的 奖励
                        score = scores_.sum(dim=1).to(device)
                        baseline_score = baseline_scores_.sum(dim=1).to(device)
                    
                                

                    
                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)
                    baseline_scores.append(baseline_score)
                    

                # 组合所有 rollout 小 batch，还原成一次 PPO update 使用的完整 batch。
                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)
                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)
                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                baseline_scores = torch.cat(baseline_scores, 0)
                
                del (logprob, ref_logprob, score, baseline_score)
                torch.cuda.empty_cache()
                gc.collect()

                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                # padding_mask=True 表示这个位置已经超过真实回答长度，不应贡献 loss。
                # 若 sequence_length=2，则索引 0、1、2 有效，3 及以后被 mask。
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)

                """
                计算 优势, 返回 最终 的 奖励
                """
                # 计算 ReMax 相对奖励和参考模型 KL 惩罚。
                # 例：随机答案 +1、贪心答案 -1，则 reward_scores=2。
                kl = logprobs - ref_logprobs
                reward_scores = scores - baseline_scores
                reward_scores = torch.clamp(reward_scores, -args.reward_clip_range, args.reward_clip_range)
                # 例：policy logp=-0.2、ref logp=-0.4，则 kl=0.2；当 kl_coef=0.05，
                # 该 token 的 kl_reward=-0.01，表示偏离参考模型需要付出代价。
                kl_reward = -args.kl_coef * kl
                returns = self.compute_returns(kl_reward, reward_scores, sequence_lengths)

                # 这里没有额外训练 value model，也没有做 advantage normalization；
                # token-level returns 直接作为 policy gradient 的 advantage。
                advantages = returns
                torch.cuda.empty_cache()

            # rollout 已经固定，下面开始优化 policy。num_ppo_epochs 表示同一批经验
            # 重复利用几次；每次先随机打乱样本，再切成 mini/micro batch。
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]

                            # 取出本次 micro batch 的固定旧经验。
                            mb_advantage = advantages[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            mb_query_responses = query_responses[micro_batch_inds]
                            mb_logprobs = logprobs[micro_batch_inds]

                            # 用更新中的 policy 重新前向，计算相同回答在“新策略”下的概率。
                            output = forward(model, mb_query_responses, processing_class.pad_token_id)
                            logits = output.logits[:, context_length - 1 : -1]
                            logits /= args.temperature + 1e-7

                            # new_logprobs 与 rollout 时保存的 mb_logprobs 分别代表新、旧策略。
                            new_logprobs = selective_log_softmax(logits, mb_responses)
                            new_logprobs = torch.masked_fill(new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB)

                            
                            """
                            计算 重要性 权重 , token 粒度
                            """
                            # PPO 概率比 ratio = exp(new_logp - old_logp)。
                            # 例：new=-0.1、old=-0.2，则 ratio=exp(0.1)≈1.105。
                            new_ratio = (new_logprobs - mb_logprobs).exp()
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)

                            # PPO clipped loss 限制一次更新幅度。cliprange=0.2 时，ratio 会在
                            # 备用目标中被约束到 [0.8, 1.2]，取两种 loss 中更保守的一项。
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = pg_loss_max.mean()
                            
                            
                            # 如果想对照最基础的 REINFORCE，可使用下面两行替代 PPO loss；
                            # 当前脚本保留注释，仅作为公式参考，不会执行。
                            # pg_losses = -new_logprobs * mb_advantage
                            # pg_loss = pg_losses.mean()

                            # Final loss
                            loss = pg_loss

                            # Accelerate 在 accumulate 上下文中负责延迟梯度同步；达到指定
                            # 累积步数后，包装过的 optimizer 才执行真正的参数更新。
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()
                            
                            # 记录策略熵、近似 KL、裁剪比例等诊断指标。
                            # clipfrac 持续过高通常意味着学习率过大或更新过激。
                            with torch.no_grad():
                                pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                                prob_dist = torch.nn.functional.softmax(logits, dim=-1)
                                entropy = torch.logsumexp(logits, dim=-1) - torch.sum(prob_dist * logits, dim=-1)
                                approxkl = 0.5 * (logprobs_diff**2).mean()
                                approxkl_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = approxkl
                                pg_clipfrac_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = (
                                    pg_clipfrac
                                )
                                pg_loss_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = pg_loss
                                entropy_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = entropy.mean()
                                ratio_stats[ppo_epoch_idx, minibatch_idx, gradient_accumulation_idx] = new_ratio.mean()
                        gradient_accumulation_idx += 1
                    minibatch_idx += 1
                
                    # 删除不再使用的大张量并释放 CUDA cache，降低下一轮的显存峰值。
                    del (
                        output, logits, new_logprobs, logprobs_diff, ratio, pg_losses,
                        pg_losses2, pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl,
                        mb_advantage, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()
                
            # 汇总一次 update 的指标；gather_for_metrics 会收集所有进程的数据。
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = kl_reward.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather_for_metrics(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = (
                    self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
                )
                metrics["objective/rlhf_reward"] = self.accelerator.gather_for_metrics(returns).mean().item()
                metrics["objective/scores"] = self.accelerator.gather_for_metrics(scores.mean()).mean().item()
                metrics["policy/approxkl_avg"] = self.accelerator.gather_for_metrics(approxkl_stats).mean().item()
                metrics["policy/clipfrac_avg"] = self.accelerator.gather_for_metrics(pg_clipfrac_stats).mean().item()
                metrics["loss/policy_avg"] = self.accelerator.gather_for_metrics(pg_loss_stats).mean().item()
                metrics["val/clipfrac_avg"] = self.accelerator.gather_for_metrics(vf_clipfrac_stats).mean().item()
                metrics["policy/entropy_avg"] = self.accelerator.gather_for_metrics(entropy_stats).mean().item()
                metrics["val/ratio"] = self.accelerator.gather_for_metrics(ratio_stats).mean().item()
                metrics["val/ratio_var"] = self.accelerator.gather_for_metrics(ratio_stats).var().item()
                metrics["val/num_eos_tokens"] = (responses == processing_class.eos_token_id).sum().item()
                metrics["lr"] = self.lr_scheduler.get_last_lr()[0]
                metrics["episode"] = self.state.episode
                self.state.epoch = self.state.episode / (self.train_dataset_len)
                self.log(metrics)
            del kl, mean_kl, mean_entropy, scores

            self.lr_scheduler.step()
            self.state.global_step += 1
            self.control = self.callback_handler.on_step_end(args, self.state, self.control)
            if self.control.should_save:
                self._save_checkpoint(model, trial=None)
                self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            torch.cuda.empty_cache()
            gc.collect()
            
            
            if self.eval_dataset and args.num_sample_generations > 0 and (update - 1) % self.sample_generations_freq == 0:
                self.generate_completions(sampling=True)
            
        # HF trainer specifics
        self.control = self.callback_handler.on_train_end(args, self.state, self.control)
        if self.control.should_save:
            self._save_checkpoint(model, trial=None, metrics=None)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


    def generate_completions(self, sampling: bool = False):
        """在验证集上生成答案，并展示 query、回答和奖励。

        ``sampling=False`` 时遍历完整验证集；``sampling=True`` 时只处理第一个
        batch，适合训练期间快速抽查。结果会打印前 5 行，并按配置上报到
        Weights & Biases 或 Comet。
        """
        args = self.args
        processing_class = self.processing_class
        # 极低温度仍保留 do_sample=True，效果接近贪心，但接口继续走采样路径。
        generation_config = GenerationConfig(
            max_new_tokens=self.args.response_length,
            temperature=(0.01 + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )

        table = defaultdict(list)
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            for batch in self.eval_dataloader:
                query = batch["prompt"]
                answer = batch["answer"]
                processing_class.padding_side = "left"
                query = processing_class.apply_chat_template(query, tokenize=True, add_generation_prompt=True, return_tensors='pt', padding=True)
                with torch.no_grad():
                    context_length = query.shape[1]
                    query_response, _ = batch_generation(
                        unwrapped_model,
                        query,
                        query.shape[0],
                        processing_class.pad_token_id,
                        generation_config,
                    )
                    response = query_response[:, context_length:]
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )
                    table["query"].extend(
                        gather_object(processing_class.batch_decode(query, skip_special_tokens=True))
                    )
                    table["model response"].extend(
                        gather_object(processing_class.batch_decode(postprocessed_response))
                    )

                    # 神经网络奖励需要完整的“prompt + response”；Python 奖励函数则在
                    # 下面解码 response 后调用，与训练阶段的奖励计算保持一致。
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)

                    if isinstance(self.reward_model, nn.Module):
                        _, score, _ = get_reward(
                            self.reward_model,
                            postprocessed_query_response,
                            processing_class.pad_token_id,
                            context_length,
                        )
                        
                    elif isinstance(self.reward_model, list):
                        scores_ = torch.zeros((query.shape[0], len(self.reward_model)))
                        
                        for i, rm in enumerate(self.reward_model):
                            if isinstance(rm, nn.Module):
                                _, score, _ = get_reward(
                            rm, postprocessed_query_response, processing_class.pad_token_id, context_length
                        )
                                scores_[:, i] = score
                            else:
                                response_text = processing_class.batch_decode(postprocessed_response, skip_special_tokens=True)
                                
                                scores_[:, i] = torch.tensor(rm(completions=response_text, answer=answer))
                                
                    
                        score = scores_.sum(dim=1).to(postprocessed_query_response.device)

                    table["score"].extend(self.accelerator.gather_for_metrics(score).float().cpu().numpy())

                if sampling:
                    break
        df = pd.DataFrame(table)

        if self.accelerator.is_main_process:
            print_rich_table(df.iloc[0 : 0 + 5])
            if "wandb" in args.report_to:
                import wandb

                if wandb.run is not None:
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            if "comet_ml" in args.report_to:
                log_table_to_comet_experiment(
                    name="completions.csv",
                    table=df,
                )


    def compute_returns(self, kl, reward_score, sequence_lengths):
        """把序列级 ReMax 奖励分配到回答中的每个有效 token。

        当前实现对终局奖励逐步乘 ``gamma``，并在每个位置加上“当前位置”的
        KL reward。它不会累加未来位置的 KL reward。

        数值示例（严格对应当前代码）：

        - ``reward_score = 2``
        - ``gamma = 0.95``
        - 三个位置的 ``kl = [-0.01, -0.02, -0.03]``

        从后向前计算得到：

        - 位置 2：``-0.03 + 2 * 0.95 = 1.87``
        - 位置 1：``-0.02 + 2 * 0.95^2 = 1.785``
        - 位置 0：``-0.01 + 2 * 0.95^3 = 1.70475``

        参数 shape：
            ``kl`` 为 ``[batch, response_length]``；``reward_score`` 和
            ``sequence_lengths`` 为 ``[batch]``。返回值与 ``kl`` shape 相同，
            padding 位置保持为 0。
        """
        returns = torch.zeros_like(kl)  # (batch_size, sequence_length)
        
        batch_size = kl.shape[0]

        for j in range(batch_size):
            # reward_score 是整条回答的奖励，在生成结束后才能确定。
            cumulative_reward = reward_score[j]
            cumulative_kl = 0
            # 仅遍历真实回答 token；padding 区域保持初始化时的 0。
            for i in reversed(range(sequence_lengths[j])):
                # 变量名沿用 cumulative_kl，但当前公式只取当前位置的 KL reward。
                cumulative_kl = kl[j, i]

                cumulative_reward *= self.args.gamma
                returns[j, i] += cumulative_kl + cumulative_reward   
        return returns


if __name__ == "__main__":

    # 模型目录既可以是本地路径，也可以是 Hugging Face Hub 模型名。
    # bfloat16 相比 float32 大约节省一半参数/激活显存，需要 GPU 支持 BF16。
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen2.5-7B-Instruct",
        torch_dtype=torch.bfloat16,  # 或者 torch.bfloat16
        trust_remote_code=True,
        # 如果需要4bit量化，可以添加以下参数：
        # load_in_4bit=True,
        # bnb_4bit_use_double_quant=True,
        # bnb_4bit_quant_type="nf4",
        # bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen2.5-7B-Instruct",
        trust_remote_code=True
    )

    # 配置 LoRA：只训练注入到线性投影层中的低秩矩阵，基座权重保持冻结。
    # q/k/v/o_proj 属于注意力层，gate/up/down_proj 属于 Qwen 的 MLP 层。
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # 将 LoRA adapter 注入基座模型。此后的 model 就是待训练 policy。
    model = get_peft_model(model, lora_config)

    # 打印可训练参数比例，用于确认没有意外开启全量参数训练。
    model.print_trainable_parameters()


    # 一次 update 的本地 rollout 样本数由以下参数共同决定：
    # per_device_train_batch_size(1) * gradient_accumulation_steps(8)
    # * num_mini_batches(父类默认值)。在单卡且 num_mini_batches=1 时为 8 条。
    training_args = ReMaxConfig(
        output_dir="output_remax",
        do_train=True,
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "paged_adamw_32bit",# 可使用paged_adamw_8bit进一步降低显存
        logging_steps = 1,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        bf16=True,
        response_length = 200,  # 每个回答最多新生成 200 个 token
        num_train_epochs=10,
        save_steps = 100,
        save_total_limit=5,
        temperature=1.0,
        stop_token_id=tokenizer.eos_token_id,
        report_to = "tensorboard"
    )
    # 数据集每条记录必须至少包含 prompt 和 answer 两列。
    # 示例：{"prompt": [{"role": "user", "content": "1+1=?"}], "answer": "2"}
    # 如果读取 Notebook 生成的 parquet，可改用：
    # datasets.load_dataset("parquet", data_files="data_gsm8k/gsm8k_train.parquet")["train"]
    dataset = datasets.load_dataset("data")['train']
 
    # ref_policy 是 policy 在训练开始前的独立副本。训练过程中 optimizer 只持有
    # policy 参数，因此参考概率保持固定，用来约束 policy 不要漂移得过远。
    trainer = ReMaxTrainer(
        config = training_args,
        policy = model,
        ref_policy = copy.deepcopy(model),
        processing_class = tokenizer,
        reward_model = [correctness_reward],
        train_dataset = dataset,
        data_collator=DataCollator()
    )
    trainer.train()
