from trl import RLOOTrainer, RLOOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
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

from trl.trainer.utils import (OnlineTrainerState,
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
# padding 部分的 log-prob 不参与真正的奖励计算。
# 这里同时填充 policy/ref_policy 的 log-prob，因此相减后 padding 位置的 KL 为 0。
INVALID_LOGPROB = 1.0

lora_rank = 32


def extract_answer(text):
    """从模型输出中提取 <answer>...</answer> 中的答案。

    例子：'<think>2+2=4</think><answer>4</answer>' -> '4'。
    """
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def correctness_reward(prompts=None, completions=None, answer=None, **kwargs):
    """GSM8K 的精确匹配奖励：正确为 +1，错误为 -1。

    例：'<answer>72</answer>' 与答案 72 匹配；
    '<answer>72.0</answer>' 与答案 72 不匹配，因为此处没有数值归一化。
    """
    responses = [extract_answer(completion) for completion in completions]
    print(f"模型输出：{completions[0]}")
    return [1 if str(response)==str(ans) else -1 for response, ans in zip(responses, answer)]

class DataCollator:
    """只保留 prompt/answer；聊天消息的 token 化和 padding 在 train() 中完成。"""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {"prompt":[feature['prompt'] for feature in features], "answer":[feature['answer'] for feature in features]}
        return batch


class MyRLOOTrainer(RLOOTrainer):

    # 重写 TRL 的 train：父类负责初始化 optimizer、dataloader、accelerator；
    # 本类实现“采样 -> 奖励 -> RLOO advantage -> PPO 更新”的主循环。
    def train(self):
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

        # 将 dataloader 封装成无限生成器；一个 epoch 结束后自动从头遍历。
        def repeat_generator():
            while True:
                yield from dataloader

        iter_dataloader = iter(repeat_generator())


        # rollout 生成参数：do_sample=True，表示对同一个 prompt 采样多个不同回答。
        # 例：response_length=200 代表每个回答最多生成 200 个 token。
        generation_config = GenerationConfig(
            max_new_tokens=args.response_length,
            temperature=(args.temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True
        )

        accelerator.print("===training policy===")
        # 记录中间过程的变量
        start_time = time.time()
        # 为每个 PPO epoch / minibatch / microbatch 预留一格统计指标。
        stats_shape = (args.num_ppo_epochs, args.num_mini_batches, args.gradient_accumulation_steps)
        approxkl_stats = torch.zeros(stats_shape, device=device)
        pg_clipfrac_stats = torch.zeros(stats_shape, device=device)
        pg_loss_stats = torch.zeros(stats_shape, device=device)
        vf_clipfrac_stats = torch.zeros(stats_shape, device=device)
        entropy_stats = torch.zeros(stats_shape, device=device)
        ratio_stats = torch.zeros(stats_shape, device=device)

        model.train()

        # 打印 日志 或者 保存 日志的 参数
        # Trainer 状态初始化。episode 统计 rollout 数；global_step 统计外层 update 数。
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

        # 生成经验 的流程
        for update in range(1, args.num_total_batches + 1):
            self.state.episode += 1 * args.batch_size
            data = next(iter_dataloader) # 一个 batch 的数据
            with torch.no_grad():

                queries = data["prompt"]
                answers = data["answer"]

                processing_class.padding_side = "left"
                # prompt 是 system/user 消息列表；在这里应用 chat template 并左侧 padding。
                # 必须是 左 padding，因为 是 自回归 模型
                queries = processing_class.apply_chat_template(queries, tokenize=True, add_generation_prompt=True, return_tensors='pt', padding=True)
                queries = queries.to(device)
                
                # RLOO：同一 prompt 采样 K 个回答。
                # 例：原始 batch 有 B=8 个问题、rloo_k=4，则 rollout batch 有 B*K=32 条回答。
                queries = queries.repeat(args.rloo_k, 1)
                # answers 也重复 rloo_k 次
                answers = answers * args.rloo_k

                context_length = queries.shape[1] # prompt 的长度
                # 按 rollout 子 batch 暂存结果，最后沿 batch 维度拼接。
                responses = []
                postprocessed_responses = []
                logprobs = []
                ref_logprobs = []
                scores = []
                sequence_lengths = []

                """
                生成 响应 的 过程 query_responses, logitss
                """
                # 用当前 policy 采样回答，并保存“采样时”的 token log-prob。
                # 这份 logprob 后续是 PPO 中固定的 old policy logprob。
                with unwrap_model_for_generation(
                    self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
                ) as unwrapped_model:
                    query_responses, logitss = batch_generation(
                        unwrapped_model,
                        queries,
                        args.local_rollout_forward_batch_size,
                        processing_class.pad_token_id,
                        generation_config,
                    )

                # 一次处理所有 B*K rollout 可能显存不足，因此按小 batch 处理。
                for i in range(0, queries.shape[0], args.local_rollout_forward_batch_size):
                  
                    query = queries[i : i + args.local_rollout_forward_batch_size]
                    answer = answers[i : i + args.local_rollout_forward_batch_size]
                    query_response = query_responses[i : i + args.local_rollout_forward_batch_size]


                    """
                    policy 模型 的 log-prob / log概率
                    """
                    response = query_response[:, context_length:] # 响应的 token，不包含 prompt                    
                    logits = logitss[i : i + args.local_rollout_forward_batch_size]
                    # 只抽取实际生成 token 的 log-prob，而不保存完整词表 softmax。
                    logprob = selective_log_softmax(logits, response)
                    del logits
                    torch.cuda.empty_cache()

                    """
                    reference 模型 的 log-prob / log概率
                    """
                    # reference policy 是冻结副本，只用于计算 KL，限制策略偏离初始策略过远。
                    ref_output = forward(ref_policy, query_response, processing_class.pad_token_id)
                    ref_logits = ref_output.logits[:, context_length - 1 : -1]
                    ref_logits /= args.temperature + 1e-7
                    ref_logprob = selective_log_softmax(ref_logits, response)
                    del ref_output, ref_logits
                    torch.cuda.empty_cache()

                    # 处理 1：EOS/stop token 后的内容截断，不计入奖励。
                    postprocessed_response = response
                    if args.stop_token_id is not None:  # handle the edge case when stop_token_id exists but is 0
                        postprocessed_response = truncate_response(
                            args.stop_token_id, processing_class.pad_token_id, response
                        )
                    
                    # 处理 2：在截断后的回答上计算任务奖励。
                    # correctness_reward 会提取 <answer> 标签并与标准答案精确比较。
                    postprocessed_query_response = torch.cat((query, postprocessed_response), 1)
                    sequence_length = first_true_indices(postprocessed_response == processing_class.pad_token_id) - 1


                    """
                    相比于 原始 的 RLOOTrainer 修改的 内容
                    """
                    if isinstance(reward_model, nn.Module): # 基于模型的 奖励
                        _, score, _ = get_reward(
                            reward_model, postprocessed_query_response, processing_class.pad_token_id, context_length
                        )
                    # else: # 基于 规则 的奖励
                    #     score = torch.tensor(
                    #         reward_model(
                    #             processing_class.batch_decode(postprocessed_query_response, skip_special_tokens=True)
                    #         ),
                    #         dtype=torch.float,
                    #     ).to(device)
                    elif isinstance(reward_model, list): # 奖励 函数/模型 是个 列表的话
                        scores_ = torch.zeros((query.shape[0], len(reward_model)))
                        
                        for i, rm in enumerate(reward_model): 
                            if isinstance(rm, nn.Module):# 基于模型 的 奖励
                                _, score, _ = get_reward(rm, postprocessed_query_response, processing_class.pad_token_id, context_length)
                                scores_[:, i] = score
                            else: # 基于规则 的奖励
                                response_text = processing_class.batch_decode(postprocessed_response, skip_special_tokens=True) # 转成 文本
                                scores_[:, i] = torch.tensor(rm(completions=response_text, answer=answer))
                                
                        # 一个句子 的 总的 奖励( 可以 在这里 设计 不同 奖励 函数的权重)
                        score = scores_.sum(dim=1).to(device)
                    
                    # 暂存当前 rollout 子 batch 的结果。
                    responses.append(response)
                    postprocessed_responses.append(postprocessed_response)
                    logprobs.append(logprob)
                    ref_logprobs.append(ref_logprob)
                    sequence_lengths.append(sequence_length)
                    scores.append(score)

                # 拼接成完整的 B*K rollout batch。
                responses = torch.cat(responses, 0)
                postprocessed_responses = torch.cat(postprocessed_responses, 0)

                logprobs = torch.cat(logprobs, 0)
                ref_logprobs = torch.cat(ref_logprobs, 0)

                sequence_lengths = torch.cat(sequence_lengths, 0)
                scores = torch.cat(scores, 0)
                
                del (logprob, ref_logprob, score)
                torch.cuda.empty_cache()
                gc.collect()

                # 处理 3：检查回答是否包含 EOS。
                # 没有 EOS 通常表示回答没有正常结束；可通过 missing_eos_penalty 额外扣分。
                contain_eos_token = torch.any(postprocessed_responses == processing_class.eos_token_id, dim=-1)
                if args.missing_eos_penalty is not None:
                    scores[~contain_eos_token] -= self.args.missing_eos_penalty
                # accelerator.print(f"{scores=}, {(contain_eos_token.sum() / len(contain_eos_token))=}")

                # padding_mask=True 的位置是 padding，不应计入 response 的 KL 或 log-prob。
                # 例：回答长度为 [3, 5] 时，第一个样本的第 4、5 个 token 是 padding。
                response_idxs = torch.arange(responses.shape[1], device=responses.device).repeat(responses.shape[0], 1)
                padding_mask = response_idxs > sequence_lengths.unsqueeze(1)
                logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)
                ref_logprobs = torch.masked_fill(ref_logprobs, padding_mask, INVALID_LOGPROB)


                """
                计算 优势 的 部分
                """
                # 4. 计算奖励。
                # kl[t] = log pi_old(token_t) - log pi_ref(token_t)，即采样策略与参考策略的差。
                kl = logprobs - ref_logprobs

                # 可选：标准化任务奖励，避免奖励尺度过大或过小导致训练不稳定。
                if args.normalize_reward:
                    scores = (scores - scores.mean()) / (scores.std() + 1e-8)
                    scores = torch.clamp(scores, -args.reward_clip_range, args.reward_clip_range)

                # 将任务奖励与 KL 惩罚合并。
                if args.token_level_kl:
                    """
                    token-level KL：每个 token 都有 KL 散度
                    """
                    # token-level KL：每个 token 都有 KL 惩罚；任务奖励放在最后一个有效 token。
                    kl_reward = -args.kl_coef * kl

                    # 找出每条回答最后一个非 padding token 的位置。
                    eos_indices = padding_mask.size(1) - 1 - padding_mask.long().fliplr().argmax(dim=1, keepdim=True)
                    last_reward = torch.zeros_like(kl)
                    # Ensure scores has correct shape and type
                    scores_shaped = scores.reshape(-1, 1).to(kl.dtype)
                    last_reward.scatter_(dim=1, index=eos_indices, src=scores_shaped)

                    # 例：任务奖励 +1、回答长度 3，则 +1 只写入第 3 个 token，
                    # 其他 token 仅承担各自的 KL 惩罚。
                    non_score_reward = kl_reward.sum(1)  # Keep this for logging
                    reward = last_reward + kl_reward
                    rlhf_reward = reward.sum(1)  # Sum across sequence length
                else:
                    """
                    序列粒度 的 KL 散度
                    """
                    # sequence-level KL：先把整条回答的 KL 求和，再与序列任务奖励相加。
                    # 例：scores=1、sequence_kl=2、kl_coef=0.1，最终奖励=1-0.2=0.8。
                    sequence_kl = kl.sum(1)
                    non_score_reward = -args.kl_coef * sequence_kl
                    """
                    KL 散度 和 任务奖励 组合 作为 最终 的 reward
                    """
                    rlhf_reward = non_score_reward + scores

                # RLOO leave-one-out advantage。
                # 例：同一 prompt 的 4 个最终奖励为 [1, 1, -1, 1]：
                # 第三个回答 baseline=(1+1+1)/3=1，advantage=-1-1=-2；
                # 第一个回答 baseline=(1-1+1)/3=1/3，advantage=1-1/3=2/3。
                # 每个回答都只与“同一 prompt 的其他回答”比较，而不是与全局 batch 比较。
                """
                优势 计算

                [
                    prompt_1, prompt_2,   # 第 1 次采样
                    prompt_1, prompt_2,   # 第 2 次采样
                    prompt_1, prompt_2,   # 第 3 次采样
                    prompt_1, prompt_2    # 第 4 次采样
                ]
                [8] -> [4, 2]
                具体看笔记
                """
                rlhf_reward = rlhf_reward.reshape(args.rloo_k, -1)
                baseline = (rlhf_reward.sum(0) - rlhf_reward) / (args.rloo_k - 1) # sum(0) 表示沿第 0 维求和，也就是把同一个 prompt 的 4 个回答奖励加起来
                advantages = rlhf_reward - baseline
                advantages = advantages.flatten()
                

                # 可选：把 advantage 标准化到均值约 0、标准差约 1，稳定 PPO 梯度尺度。
                if args.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                torch.cuda.empty_cache()

            # 对同一批 rollout 执行多个 PPO epoch；每个 epoch 重新打乱样本，
            # 但 advantage 和采样时保存的 old logprob 保持固定。
            """
            根据 经验 计算 梯度 更新 模型
            """
            for ppo_epoch_idx in range(args.num_ppo_epochs):
                b_inds = np.random.permutation(args.local_batch_size)
                minibatch_idx = 0 # 减少 显存占用
                for mini_batch_start in range(0, args.local_batch_size, args.local_mini_batch_size):
                    mini_batch_end = mini_batch_start + args.local_mini_batch_size
                    mini_batch_inds = b_inds[mini_batch_start:mini_batch_end]
                    gradient_accumulation_idx = 0
                    for micro_batch_start in range(0, args.local_mini_batch_size, args.per_device_train_batch_size):
                        with accelerator.accumulate(model):
                            micro_batch_end = micro_batch_start + args.per_device_train_batch_size
                            micro_batch_inds = mini_batch_inds[micro_batch_start:micro_batch_end]

                            # 取当前 microbatch 的 advantage、回答、完整 query+response，
                            # 以及采样时保存的 old policy logprob。
                            mb_advantage = advantages[micro_batch_inds]
                            mb_responses = responses[micro_batch_inds]
                            mb_query_responses = query_responses[micro_batch_inds]
                            """
                            旧策略
                            """
                            mb_logprobs = logprobs[micro_batch_inds]


                            """
                            当前策略
                            """
                            # 当前 policy 对已经采样出的回答重新前向计算。
                            output = forward(model, mb_query_responses, processing_class.pad_token_id)
                            logits = output.logits[:, context_length - 1 : -1]
                            logits /= args.temperature + 1e-7
                            # 当前 policy 对相同 response token 的新 logprob。
                            new_logprobs = selective_log_softmax(logits, mb_responses)
                            new_logprobs = torch.masked_fill(new_logprobs, padding_mask[micro_batch_inds], INVALID_LOGPROB)

                                
                            # 计算 PPO 的序列概率比。
                            # 先逐 token 计算 ratio 供统计，再把 token logprob 求和，
                            # 得到完整回答级别的 ratio。
                            # 例：old 序列概率=0.4、new 序列概率=0.6，则 ratio=1.5；
                            # cliprange=0.2 时，正 advantage 的有效 ratio 最多按 1.2 计算。
                            new_ratio = (new_logprobs - mb_logprobs).exp()
                            """
                            求和 token 级 -> 序列级(句子粒度)
                            """
                            new_logprobs = new_logprobs.sum(1)
                            mb_logprobs = mb_logprobs.sum(1)
                            logprobs_diff = new_logprobs - mb_logprobs
                            ratio = torch.exp(logprobs_diff)

                            """
                            PPO loss
                            """
                            # PPO clipped loss（采用“最小化 loss”的写法）。
                            # 标准最大化目标是 min(r*A, clip(r)*A)，
                            # 取负号后等价于 max(-r*A, -clip(r)*A)。
                            pg_losses = -mb_advantage * ratio
                            pg_losses2 = -mb_advantage * torch.clamp(ratio, 1.0 - args.cliprange, 1.0 + args.cliprange)
                            pg_loss_max = torch.max(pg_losses, pg_losses2)
                            pg_loss = pg_loss_max.mean()
                            
                            
                            # 如果改用 REINFORCE loss，可以取消下面两行注释。
                            # L_REINFORCE = -A * log pi_new(a|s)。在 on-policy、ratio=1、
                            # 且没有触发 clipping 时，PPO surrogate 与 REINFORCE 的反向梯度相同。
                            # 当 ratio 偏离 1 或触发 clipping 后，两者不再等价。
                            # pg_losses = -new_logprobs * mb_advantage
                            # pg_loss = pg_losses.mean()

                            # Final loss
                            loss = pg_loss

                            # 反向传播和参数更新；实际可训练参数主要是 LoRA 参数。
                            accelerator.backward(loss)
                            optimizer.step()
                            optimizer.zero_grad()

                            with torch.no_grad():
                                pg_clipfrac = (pg_losses2 > pg_losses).float().mean()
                                # entropy 越高表示 token 分布越随机；过低可能意味着策略过早坍缩。
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

                    # del everything and empty cache
                    # fmt: off
                    del (
                        output, logits, new_logprobs, logprobs_diff, ratio, pg_losses,
                        pg_losses2, pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl,
                        mb_advantage, mb_responses, mb_query_responses, mb_logprobs,
                    )
                    # fmt: on
                    torch.cuda.empty_cache()

            # 记录本轮 rollout 和 PPO 更新的统计量。
            with torch.no_grad():
                mean_kl = kl.sum(1).mean()
                mean_entropy = (-logprobs).sum(1).mean()
                mean_non_score_reward = non_score_reward.mean()
                eps = int(self.state.episode / (time.time() - start_time))
                metrics = {}
                metrics["eps"] = eps
                metrics["objective/kl"] = self.accelerator.gather_for_metrics(mean_kl).mean().item()
                metrics["objective/entropy"] = self.accelerator.gather_for_metrics(mean_entropy).mean().item()
                metrics["objective/non_score_reward"] = (
                    self.accelerator.gather_for_metrics(mean_non_score_reward).mean().item()
                )
                metrics["objective/rlhf_reward"] = self.accelerator.gather_for_metrics(rlhf_reward).mean().item()
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
                self.state.epoch = self.state.episode / (args.rloo_k * self.train_dataset_len)  # used by self.log
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
        """
        在 eval_dataset 上生成回答并打印 reward 表格。

        sampling=True 时只展示一个 batch，适合训练中周期性快速抽样；
        sampling=False 时遍历完整的 eval_dataloader。
        """
        args = self.args
        processing_class = self.processing_class
        # do_sample=True 但 temperature 很低，因此行为接近贪心生成。
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



if __name__ == "__main__":
    # 若模型不是当前目录下的本地路径，通常应使用完整的 Hugging Face 名称，
    # 例如 "Qwen/Qwen2.5-7B-Instruct"。
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

    # 配置 LoRA：冻结基础模型，只训练插入 attention/MLP 投影层的低秩矩阵。
    # r=32 是低秩矩阵 rank；rank 越大，表达能力和显存/计算开销通常越高。
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

    # 应用 LoRA，并打印可训练参数比例。
    model = get_peft_model(model, lora_config)

    # 打印模型信息
    model.print_trainable_parameters()


    # 例：per_device_train_batch_size=1、gradient_accumulation_steps=8，
    # 表示使用 8 次梯度累积扩大有效训练 batch；rloo_k=4 表示每题采样 4 个回答。
    training_args = RLOOConfig(
        output_dir="output_rloo",
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
        length_column_name='prompt',
        response_length = 200,
        num_train_epochs=10,
        save_steps = 200,
        save_total_limit=5,
        temperature=1.0,
        stop_token_id=tokenizer.eos_token_id,
        rloo_k=4,
        report_to = "tensorboard"
    )
    # 数据集至少需要包含 prompt（聊天消息列表）和 answer（标准答案）字段。
    # 注意：data_process.ipynb 当前保存到 data_gsm8k，需保证此路径与实际数据一致。
    dataset = datasets.load_dataset("./data")['train']
 
    trainer = MyRLOOTrainer(
        config = training_args,
        policy = model,
        # reference policy 只用于 KL 约束，不参加 optimizer 更新；deepcopy 会额外占用模型显存。
        ref_policy = copy.deepcopy(model),
        processing_class = tokenizer,
        reward_model = [correctness_reward],
        train_dataset = dataset,
        data_collator=DataCollator()
    )
    trainer.train()
