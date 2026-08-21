"""使用不同 tokenizer 的教师模型和学生模型进行知识蒸馏。

端到端流程示例：

1. 数据集提供 ``prompt="你是谁？"``、``answer="我是AI。"``；
2. 学生 tokenizer 按 Qwen chat template 编码，教师 tokenizer 按 GLM chat
   template 编码，因此两边的 token 数量和 token id 通常不同；
3. 教师和学生分别前向计算 logits；
4. ``ULDLoss`` 先按解码文本对齐答案位置，再对齐两边不同大小的词表；
5. 只反向更新带 LoRA adapter 的学生模型，教师模型始终处于 eval/no_grad。

常用形状记号：``B`` 表示 batch size，``Ls/Lt`` 表示学生/教师序列长度，
``Vs/Vt`` 表示学生/教师词表大小。
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import Trainer, TrainingArguments
from dataset import SFTDataset,MyDataCollator


class ULDLoss(nn.Module):
    """
    计算跨 tokenizer 的混合 ULD 蒸馏损失。

    损失由两部分组成::

        total_loss = crossentropy_weight * CE
                   + distillation_weight * ULD

    ULD 又分为：

    * 两个词表中字符串完全相同的 token：使用 KL 散度；
    * 无法按 token 字符串匹配的部分：概率降序排列、补零后使用 L1 距离。

    默认 ``crossentropy_weight=0``、``distillation_weight=1``，即当前训练入口
    实际执行纯蒸馏，不使用数据答案的标准交叉熵作为额外监督。

    ``skip_eos`` 控制蒸馏时是否排除答案末尾的 EOS。它不影响数据集中的
    answer 文本，只影响这里选取的答案位置数。
    """

    def __init__(self, student_tokenizer=None, teacher_tokenizer=None, crossentropy_weight=0.0, distillation_weight=1.0, temperature=1, skip_eos=False):
        super().__init__()

        # 学生模型 和 教师 模型 的 tokenizer
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer

        # 损失权重
        self.crossentropy_weight = crossentropy_weight
        self.distillation_weight = distillation_weight

        self.temperature = temperature
        self.skip_eos = skip_eos # 是否忽略 结束标识符 EOS

        vocab_mapping, teacher_matched_ids, student_matched_ids = self.init_vocab_mapping()

        # vocab_mapping 的方向是 teacher_id -> student_id。
        # 例：两边词表都存在字符串 "hello"，其 id 分别为 100 和 200，
        # 则 vocab_mapping[100] == 200。
        self.vocab_mapping = vocab_mapping
        self.teacher_matched_ids = teacher_matched_ids
        self.student_matched_ids = student_matched_ids

    def __call__(
        self, student_logits, teacher_logits, student_labels, teacher_labels, student_input_ids, teacher_input_ids
    ):
        """组合监督交叉熵与蒸馏损失。

        主要输入形状示例::

            student_logits:    [B, Ls, Vs]
            teacher_logits:    [B, Lt, Vt]
            student_labels:    [B, Ls]
            teacher_labels:    [B, Lt]
            student_input_ids: [B, Ls]
            teacher_input_ids: [B, Lt]

        提示：``nn.Module`` 通常实现 ``forward`` 而不是直接覆盖 ``__call__``；
        这里保留项目原有写法，仅解释当前行为。
        """

        # 交叉熵 损失
        if self.crossentropy_weight > 0:
            # CausalLM 的 logits[t] 用于预测 input_ids[t+1]，所以标准 CE 需要
            # logits 去掉最后一位、labels 去掉第一位后再配对。
            # 例：logits 的位置 [0,1,2] 分别监督 labels 的位置 [1,2,3]。
            shift_logits = student_logits[..., :-1, :].contiguous()
            shift_labels = student_labels[..., 1:].contiguous()
            # prompt 和 padding 在 labels 中都被填成 pad_token_id，因此不会
            # 参与 CE。若某 tokenizer 的 pad_id 与 eos_id 相同，EOS 也会被
            # 一并忽略，这是当前实现需要注意的行为。
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.student_tokenizer.pad_token_id)
            crossentropy_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            crossentropy_loss = self.crossentropy_weight * crossentropy_loss
        else:
            crossentropy_loss = 0.0

        # 蒸馏 损失
        distillation_loss = self.compute_distillation_loss(
            student_logits, teacher_logits, student_labels, teacher_labels, student_input_ids, teacher_input_ids
        )

        return crossentropy_loss + distillation_loss * self.distillation_weight

    def init_vocab_mapping(self):
        """
        按 tokenizer 词表中的原始 token 字符串建立一对一 id 映射。

        示例::

            teacher_vocab = {"你": 8, "好": 9}
            student_vocab = {"你": 3, "世界": 4}

        结果为 ``vocab_mapping={8: 3}``，"好" 与 "世界" 分别进入两边的
        unmatched 集合。这里比较的是词表内部字符串，不是 token 解码后的
        Unicode 文本，因此 ``"▁hello"`` 与 ``"hello"`` 不会匹配。

        vocab_mapping 映射关系是 teacher_id -> student_id。
        teacher_matched_ids 和 student_matched_ids 分别是匹配的 id 集合。
        """

        student_vocab = self.student_tokenizer.get_vocab()
        teacher_vocab = self.teacher_tokenizer.get_vocab()
        
        student_token_to_id = dict(student_vocab.items())
        vocab_mapping = {}
        
        teacher_matched_ids = set()
        student_matched_ids = set()

        for token_str, teacher_token_id in teacher_vocab.items():
            if token_str in student_token_to_id:
                student_token_id = student_token_to_id[token_str]
                vocab_mapping[teacher_token_id] = student_token_id
                teacher_matched_ids.add(teacher_token_id)
                student_matched_ids.add(student_token_id)

        return vocab_mapping, teacher_matched_ids, student_matched_ids


    def get_start_and_size_answers(self, answers, tokenizer):
        """根据非 padding 标签找出每条答案的起点和长度。

        例：``pad_token_id=0`` 且 labels 为::

            [0, 0, 21, 22, 23, 0]

        返回 ``start=2``、``size=3``。这里假设有效答案 token 连续排列，且
        序列右侧只有 padding。
        """

        answers_index = []
        answers_size = []

        for answer in answers:
            answer_mask = answer.ne(tokenizer.pad_token_id)
            if not answer_mask.any():
                # 空答案（或所有 token id 都等于 pad_id）使用 (0, 0) 作为
                # 占位结果；后续 EOS 分支可能继续调整 size。
                answers_index.append(0)
                answers_size.append(0)
                continue

            indices = answer_mask.nonzero(as_tuple=True)[0]
            answers_index.append(int(indices[0].item()))
            answers_size.append(int(answer_mask.sum().item()))
        return answers_index, answers_size
    
    
    def compute_distillation_loss(
        self, student_logits, teacher_logits, student_labels, teacher_labels, student_input_ids, teacher_input_ids
    ):
        """
        student_logits:    [B, L_sequence, Vocab_size]
        teacher_logits:    [B, L_sequence, Vocab_size]

        逐条样本完成答案序列对齐、词表对齐，再对 batch 求平均。

        学生和教师可能将同一答案切成不同长度，例如::

            原文:       "蓝色"
            学生 tokens: ["蓝", "色"]       -> 长度 2
            教师 tokens: ["蓝色"]           -> 长度 1

        本函数先将上述位置合并到同一个文本组，再调用
        ``compute_hybrid_uld_loss`` 处理 ``Vs != Vt`` 的问题。

        注意：当前代码从 ``student_start`` 直接截取 logits。对标准 causal LM
        来说，答案第一个 token 实际由 ``student_start - 1`` 位置预测，因此
        当前蒸馏 logits 与答案 token 存在一位偏移；本次仅添加注释，未修改
        原有算法。
        """

        """
        获取答案的起始位置和长度。
        为了 获取答案
        """
        student_answer_index, student_answer_size = self.get_start_and_size_answers(student_labels, self.student_tokenizer)
        teacher_answer_index, teacher_answer_size = self.get_start_and_size_answers(teacher_labels, self.teacher_tokenizer)

        # 当 eos_id != pad_id 时，get_start_and_size_answers 能直接数到 EOS；
        # skip_eos=True 时将长度减一即可。
        if self.student_tokenizer.eos_token_id != self.student_tokenizer.pad_token_id:
            if self.skip_eos:
                student_answer_size = [size - 1 for size in student_answer_size]

        # 当 eos_id == pad_id 时，EOS 会被 answer_mask 当成 padding 排除；
        # 若不跳过 EOS，需要把它重新补回答案长度。
        else:
            if not self.skip_eos:
                student_answer_size = [size + 1 for size in student_answer_size]

        if self.teacher_tokenizer.eos_token_id != self.teacher_tokenizer.pad_token_id:
            if self.skip_eos:
                teacher_answer_size = [size - 1 for size in teacher_answer_size]

        else:
            if not self.skip_eos:
                teacher_answer_size = [size + 1 for size in teacher_answer_size]


        batch_size = student_logits.size(0)
        # 一个 batch 内的 蒸馏损失
        distillation_losses = []

        # 每条样本的 tokenizer 切分和对齐组都可能不同，因此这里逐样本处理。
        for i in range(batch_size):

            # 开始 索引 和 长度
            student_start = student_answer_index[i]
            student_size = student_answer_size[i]
            teacher_start = teacher_answer_index[i]
            teacher_size = teacher_answer_size[i]

            # 只保留答案对应的时间步。两边第一维长度允许不同，但各自最后
            # 一维仍是自己的完整词表大小。
            student_answer_logits = student_logits[i, student_start : student_start + student_size] # [student_len, student_vocab_size]
            teacher_answer_logits = teacher_logits[i, teacher_start : teacher_start + teacher_size] # [teacher_len, teacher_vocab_size]

            # 温度 softmax 将模型输出转换成待比较的概率分布。
            student_probs = F.softmax(student_answer_logits / self.temperature, dim=-1)
            teacher_probs = F.softmax(teacher_answer_logits / self.temperature, dim=-1)

            # token id 只用于还原文本分片和建立位置对齐，不直接跨词表比较 id。
            student_token_ids = student_input_ids[i, student_start : student_start + student_size].tolist()  # [student_len]
            teacher_token_ids = teacher_input_ids[i, teacher_start : teacher_start + teacher_size].tolist()  # [teacher_len]
            # 针对 tokenizer 分词后长度不一致的问题，进行对齐处理。
            # 方案一（截断）
            # min_length = min(len(student_token_ids), len(teacher_token_ids))
            # student_aligned = student_probs[:min_length, :]
            # teacher_aligned = teacher_probs[:min_length, :]
            
            # 方案二（当前启用）：按解码后的文本内容分组对齐。
            if self.skip_eos:
                student_alignment_groups, teacher_alignment_groups = self.get_alignment_groups_from_ids(student_token_ids, teacher_token_ids)

                student_aligned = self.merge_prob_with_alignment_groups(student_probs, student_alignment_groups)
            
                teacher_aligned = self.merge_prob_with_alignment_groups(teacher_probs, teacher_alignment_groups)
            
            else:
                # EOS 是特殊 token，不参加普通文本分组；先对齐 EOS 之前的
                # 答案，最后再把两边的 EOS 概率位置各自追加回来。
                student_alignment_groups, teacher_alignment_groups = self.get_alignment_groups_from_ids(student_token_ids[:-1], teacher_token_ids[:-1])

                student_aligned = self.merge_prob_with_alignment_groups(student_probs[:-1, :], student_alignment_groups)
            
                teacher_aligned = self.merge_prob_with_alignment_groups(teacher_probs[:-1, :], teacher_alignment_groups)
            
                student_aligned = torch.cat([student_aligned, student_probs[-1:, :]], dim=0)
                teacher_aligned = torch.cat([teacher_aligned, teacher_probs[-1:, :]], dim=0)
        

            # 针对 vocab size 不一致的问题，进行对齐处理。
            # 方案一（不区分匹配和不匹配的token，统一处理：sort+pad）
            # student_sorted = student_aligned.sort(dim=-1, descending=True).values
            # teacher_sorted = teacher_aligned.sort(dim=-1, descending=True).values

            # student_vocab_size = student_sorted.size(-1)
            # teacher_vocab_size = teacher_sorted.size(-1)
            # max_vocab_size = max(student_vocab_size, teacher_vocab_size)

            # if student_vocab_size < max_vocab_size:
            #     student_sorted = F.pad(student_sorted, (0, max_vocab_size - student_vocab_size))
            # if teacher_vocab_size < max_vocab_size:
            #     teacher_sorted = F.pad(teacher_sorted, (0, max_vocab_size - teacher_vocab_size))

            # # Compute L1 distance (ULD approach)
            # aligned_loss = F.l1_loss(student_sorted, teacher_sorted, reduction="sum")
            # aligned_loss /= student_aligned.size(0) 
            
            # 方案二（当前启用）：匹配 token 计算 KL，未匹配 token 排序后
            # 计算 L1。aligned_loss 是当前样本的标量损失。
            aligned_loss = self.compute_hybrid_uld_loss(student_aligned, teacher_aligned)

            distillation_losses.append(aligned_loss)

        distillation_loss = torch.stack(distillation_losses).mean()
        return distillation_loss

    def get_alignment_groups_from_ids(self, student_token_ids, teacher_token_ids):
        """
        根据累计解码文本，为两种 token 切分建立位置分组。

        假设同一段文本在两边被切分为::

            student pieces = ["知", "识", "蒸馏"]
            teacher pieces = ["知识", "蒸", "馏"]

        可能得到::

            student groups = [[0, 1], [2]]
            teacher groups = [[0], [1, 2]]

        每一对 group 解码后应表示相同文本：第一组都是“知识”，第二组都是
        “蒸馏”。返回的索引是相对于答案片段的局部位置，不是完整输入位置。

        注意：实现假设 ``decode(ids[:k])`` 是 ``decode(ids[:k+1])`` 的文本
        前缀。字节级 tokenizer 遇到不完整 UTF-8 token 时不一定满足该假设，
        此时生成的 piece 可能不准确。
        """

        def to_canonical_pieces(tok, ids):
            """用“累计 decode 的新增后缀”表示每个 token 的文本贡献。

            例：累计解码结果依次是 ``"知"``、``"知识"``、``"知识蒸馏"``，
            则得到 pieces ``["知", "识", "蒸馏"]``。
            """

            pieces = []
            prev = ""
            for k in range(len(ids)):
                cur = tok.decode(ids[: k + 1], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                pieces.append(cur[len(prev) :])
                prev = cur
            return pieces

        """
        贪婪匹配 的 方法 进行 分组
        """
        s_pieces = to_canonical_pieces(self.student_tokenizer, student_token_ids)
        t_pieces = to_canonical_pieces(self.teacher_tokenizer, teacher_token_ids)

        i = j = 0
        s_buf = t_buf = ""
        s_group = []
        t_group = []
        s_groups = []
        t_groups = []

        def flush():
            # 只有学生、教师两边都消费了至少一个 token，才构成有效对齐组。
            if s_group and t_group:
                s_groups.append(s_group.copy())
                t_groups.append(t_group.copy())

        while i < len(s_pieces) or j < len(t_pieces):
            # 两边缓冲文本完全相同，说明找到了一个最小的共同文本边界。
            if s_buf == t_buf and s_buf != "":
                flush()
                s_buf = t_buf = ""
                s_group = []
                t_group = []
                continue

            if s_buf == "" and i < len(s_pieces):
                # 每轮先保证学生缓冲区至少有一个 piece。
                s_buf += s_pieces[i]
                s_group.append(i)
                i += 1
                continue
            if t_buf == "" and j < len(t_pieces):
                # 再保证教师缓冲区至少有一个 piece。
                t_buf += t_pieces[j]
                t_group.append(j)
                j += 1
                continue

            if len(s_buf) <= len(t_buf):
                # 当前学生文本更短（或等长但尚不相同），继续消费学生 token。
                if i < len(s_pieces):
                    s_buf += s_pieces[i]
                    s_group.append(i)
                    i += 1
                elif j < len(t_pieces):
                    t_buf += t_pieces[j]
                    t_group.append(j)
                    j += 1
            else:
                # 当前教师文本更短，继续消费教师 token。
                if j < len(t_pieces):
                    t_buf += t_pieces[j]
                    t_group.append(j)
                    j += 1
                elif i < len(s_pieces):
                    s_buf += s_pieces[i]
                    s_group.append(i)
                    i += 1

        if s_buf == t_buf and s_group and t_group:
            flush()
        elif s_group or t_group:
            # 如果最终累计文本仍不相同，也保留剩余 token 作为兜底组。某一边
            # 可能得到空组，merge_prob_with_alignment_groups 会把它变成全零行。
            if s_group or t_group:
                if not s_group:
                    s_group = []
                if not t_group:
                    t_group = []
                if s_group or t_group:
                    s_groups.append(s_group.copy() if s_group else [])
                    t_groups.append(t_group.copy() if t_group else [])

        return s_groups, t_groups

    def merge_prob_with_alignment_groups(self, probs, alignment_groups):
        """
        把多个 token 位置的概率合并成一个对齐位置。

        输入 ``probs`` 的形状为 ``[answer_len, vocab_size]``。例如 group 为
        ``[[0, 1], [2]]``，输出长度会从 3 变为 2：位置 0、1 合并，位置 2
        原样保留。

        多位置组使用的当前规则是：对同一 vocab id 的概率取乘积后重新归一化::

            merged[v] = normalize(probs[0, v] * probs[1, v] * ...)

        代码在 log 空间执行乘法以提高数值稳定性。这是一种实验性启发式，
        并不等同于两个连续 token 组成文本片段的严格联合概率。
        """

        if not alignment_groups:
            # 无需对齐时保持原始序列长度和概率不变。
            return probs

        vocab_size = probs.size(-1)
        target_len = len(alignment_groups)
        aligned_probs = torch.zeros(target_len, vocab_size, device=probs.device)

    
        for group_idx, group in enumerate(alignment_groups):
            if len(group) > 1:
                eps = 1e-8
                # log(a*b) = log(a) + log(b)；clamp_min 防止 log(0)。
                logp = torch.log(probs[group[0]].clamp_min(eps))
                for idx in group[1:]:
                    if idx < probs.size(0):
                        logp = logp + torch.log(probs[idx].clamp_min(eps))
                aligned_probs[group_idx] = torch.softmax(logp, dim=-1)
            elif len(group) == 1:
                aligned_probs[group_idx] = probs[group[0]]
            else:
                # 某一 tokenizer 没有与另一边剩余文本对应的 token 时，用全零
                # 概率向量占位，使学生、教师拥有相同的对齐组数量。
                aligned_probs[group_idx] = torch.zeros_like(probs[0])

        return aligned_probs

    def compute_hybrid_uld_loss(self, student_aligned, teacher_aligned):
        """
        在每个已对齐文本位置比较两个不同大小的词表分布。

        输入形状为::

            student_aligned: [aligned_len, student_vocab_size]
            teacher_aligned: [aligned_len, teacher_vocab_size]

        假设教师词表有 5 个 token、学生词表有 4 个 token，其中 3 个 token
        字符串可以一一匹配，则：

        * matched 分支比较 3 对具有相同字符串语义的概率；
        * unmatched 分支分别取教师剩余 2 维、学生剩余 1 维，降序排列后把
          较短的一边补零，再计算 L1；
        * 权重为 ``matched_weight=3/5``、``unmatched_weight=2/5``。

        排序后的 unmatched 分支只比较概率分布的“形状”，不再假设两边第 k
        个 token 具有相同语义。
        """

        device = student_aligned.device
        student_vocab_size = student_aligned.size(-1)
        teacher_vocab_size = teacher_aligned.size(-1)

        if self.teacher_matched_ids:
            # 两个 id tensor 的相同下标代表同一个 token 字符串：
            # teacher_matched_token_ids[k] <-> student_matched_token_ids[k]。
            teacher_matched_token_ids = torch.tensor(sorted(self.teacher_matched_ids), dtype=torch.long, device=device)
            student_matched_token_ids = torch.tensor([self.vocab_mapping[token_id.item()] for token_id in teacher_matched_token_ids], dtype=torch.long, device=device)
        else:
            teacher_matched_token_ids = torch.tensor([], dtype=torch.long, device=device)
            student_matched_token_ids = torch.tensor([], dtype=torch.long, device=device)

        # mask 用于过滤掉未匹配的 token 位置
        teacher_matched_mask = torch.zeros(teacher_vocab_size, dtype=torch.bool, device=device)
        student_matched_mask = torch.zeros(student_vocab_size, dtype=torch.bool, device=device)

        if len(teacher_matched_token_ids) > 0:
            # True 表示该词表位置已经有跨 tokenizer 的明确映射。
            teacher_matched_mask[teacher_matched_token_ids] = True
            student_matched_mask[student_matched_token_ids] = True

        matched_loss = torch.tensor(0.0, device=device)
        matched_token_count = 0
        if len(teacher_matched_token_ids) > 0:
            """
            重合部分 的 KL 散度
            """
            # 按映射后的相同顺序抽取概率，得到 [aligned_len, num_matched]。
            teacher_matched_probs = teacher_aligned[:, teacher_matched_token_ids]  # [seq_len, num_matched]
            student_matched_probs = student_aligned[:, student_matched_token_ids]  # [seq_len, num_matched]
            matched_token_count = teacher_matched_probs.size(-1)
            matched_loss = self.compute_kl_loss(student_matched_probs, teacher_matched_probs)

        """
        取反后，mask 选中的都是无法按 token 字符串建立映射的词表项。

        先排序
        后padding
        然后计算 L1 损失
        """
        teacher_unmatched_mask = ~teacher_matched_mask
        student_unmatched_mask = ~student_matched_mask

        teacher_unmatched_probs = teacher_aligned[:, teacher_unmatched_mask]  # [seq_len, num_teacher_unmatched]
        student_unmatched_probs = student_aligned[:, student_unmatched_mask]  # [seq_len, num_student_unmatched]

        unmatched_loss = torch.tensor(0.0, device=device)
        if teacher_unmatched_probs.size(-1) > 0 and student_unmatched_probs.size(-1) > 0:
            # 例：[0.1, 0.7, 0.2] 排序后为 [0.7, 0.2, 0.1]。排序会丢弃
            # 原 token id，只保留高、中、低概率的相对形状。
            teacher_unmatched_sorted = teacher_unmatched_probs.sort(dim=-1, descending=True).values
            student_unmatched_sorted = student_unmatched_probs.sort(dim=-1, descending=True).values

            teacher_unmatched_size = teacher_unmatched_sorted.size(-1)
            student_unmatched_size = student_unmatched_sorted.size(-1)
            max_unmatched_size = max(teacher_unmatched_size, student_unmatched_size)

            # 两边 unmatched 维数不同，较短的一侧在 vocab 维右侧补 0。
            if teacher_unmatched_size < max_unmatched_size:
                teacher_unmatched_sorted = F.pad(
                    teacher_unmatched_sorted, (0, max_unmatched_size - teacher_unmatched_size)
                )
            if student_unmatched_size < max_unmatched_size:
                student_unmatched_sorted = F.pad(
                    student_unmatched_sorted, (0, max_unmatched_size - student_unmatched_size)
                )

            unmatched_loss = F.l1_loss(student_unmatched_sorted, teacher_unmatched_sorted, reduction="sum")
            # 先对所有位置和词表维求和，再除以对齐后的序列长度。
            unmatched_loss /= student_aligned.size(0)  

        """
        加权
        """
        # 当前实现以“教师词表中有多少比例可匹配”作为两类损失的权重。
        matched_weight = matched_token_count / max(1, teacher_vocab_size)
        unmatched_weight = 1.0 - matched_weight

        total_loss = matched_weight * matched_loss + unmatched_weight * unmatched_loss

        return total_loss # 一条序列 总的 损失

    def compute_kl_loss(self, student_logits, teacher_logits):
        """计算 matched token 上的前向 KL ``KL(teacher || student)``。

        ``F.kl_div(input, target, log_target=True)`` 要求 input 和 target 都是
        log 概率，所以函数先对两边执行 ``log_softmax``。

        注意：虽然参数名为 ``*_logits``，当前调用方实际传入的已经是 softmax
        概率（见 ``compute_distillation_loss``）。因此这里会把概率除以温度后
        再做一次 log_softmax，而不是直接对原模型 logits 计算 KL；这是当前
        代码的真实行为，本次注释没有调整算法。

        ``mean()`` 会同时对序列位置和 matched vocab 维取平均。例如输入形状
        为 ``[4, 1000]``，最终对 4000 个逐元素 KL 项求平均。
        """

        batch_seq_len, num_matched = student_logits.shape

        # 当前输入本来已经是二维；view 保留了未来扩展/重排后的二维形式。
        student_logits = student_logits.view(-1, num_matched)
        teacher_logits = teacher_logits.view(-1, num_matched)

        student_logits = student_logits / self.temperature
        teacher_logits = teacher_logits / self.temperature

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        # 以教师为 target，计算 teacher -> student 的前向 KL。
        kl_loss = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)

        return kl_loss.mean()



class KGTrainer(Trainer):
    """在 Hugging Face Trainer 中同时执行学生和教师前向计算。

    每个 batch 的数据流如下::

        dataset 中的学生 token ids
                    |
                    v
        解码为 prompt/answer 文本
             ╱                 ╲
            v                   v
        学生 tokenizer       教师 tokenizer
            |                   |
        学生前向（有梯度）   教师前向（no_grad）
             ╲                 ╱
              ---- ULDLoss ----

    ``Trainer`` 只把 ``model``（学生）交给优化器；``teacher_model`` 作为附加
    属性保存，不参与反向传播和参数更新。
    """

    def __init__(
        self,
        model = None,
        teacher_model = None,
        args = None,
        data_collator = None, 
        train_dataset = None,
        eval_dataset = None,
        tokenizer = None,
        teacher_tokenizer = None,
        model_init = None, 
        compute_metrics = None, 
        callbacks = None,
        optimizers = (None, None), 
        preprocess_logits_for_metrics = None,
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer
        # 初始化时扫描一次两个完整词表，提前构造 matched token id 映射。
        self.uld_loss_fn = ULDLoss(student_tokenizer=tokenizer, teacher_tokenizer=teacher_tokenizer)


    def get_inputs_from_texts(self, tokenizer, prompt_texts: list[str], answer_texts: list[str]):
        """
        使用指定 tokenizer 构造一批标准的 CausalLM 输入。

        单条样本的逻辑示例（数字仅用于说明）::

            prompt_ids = [1, 10, 11, 12]       # 已应用 chat template
            answer_ids = [20, 21, 2]           # 最后的 2 是 eos_token_id
            sequence   = [1, 10, 11, 12, 20, 21, 2]
            labels     = [0,  0,  0,  0, 20, 21, 2]
            mask       = [1,  1,  1,  1,  1,  1, 1]

        假设 ``pad_token_id=0``，prompt 标签填 0，表示 prompt 不作为答案监督。
        batch 内较短序列在右侧继续补 pad，attention_mask 对应补 0。

        学生和教师分别调用本函数，所以即使传入文本相同，最终序列长度、
        特殊 token 和 token id 也可以不同。
        """

        sequences = []
        labels_list = []
        attention_masks = []

        for prompt_text, answer_text in zip(prompt_texts, answer_texts):
            # add_generation_prompt=True 会在对话末尾加入“轮到 assistant 回答”
            # 所需的模型专属标记。
            messages = [{'role': 'user', 'content': prompt_text}]
            prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            # prompt 已经由 chat template 生成了特殊标记。部分 tokenizer 在
            # encode 默认 add_special_tokens=True 时可能再次添加 BOS；这里是
            # 项目当前行为，使用其他模型时需要核对其 tokenizer 配置。
            prompt_ids = tokenizer.encode(prompt)
            # 答案不额外添加 BOS，只在末尾显式加入一个 EOS。
            answer_ids = tokenizer.encode(answer_text, add_special_tokens=False) + [tokenizer.eos_token_id]
            sequence = prompt_ids + answer_ids
            attention_mask = [1] * len(sequence)

            # pad_token_id 在这里同时承担“忽略 prompt 标签”的哨兵值。
            labels = [tokenizer.pad_token_id] * len(prompt_ids) + answer_ids

            # 此处先创建一维 CPU tensor；pad_sequence 会统一 batch 长度，之后
            # compute_loss 再把学生/教师 batch 移到各自模型设备。
            sequences.append(torch.tensor(sequence))
            labels_list.append(torch.tensor(labels))
            attention_masks.append(torch.tensor(attention_mask))
        
        # 右侧 padding。若原始长度为 5 和 3，输出形状为 [2, 5]。
        input_ids = pad_sequence(sequences, batch_first=True, padding_value=tokenizer.pad_token_id)
        labels = pad_sequence(labels_list, batch_first=True, padding_value=tokenizer.pad_token_id)
        attention_mask = pad_sequence(attention_masks, batch_first=True, padding_value=0)

        return input_ids, labels, attention_mask
        
    
    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        """覆盖 Trainer 的默认损失，执行跨 tokenizer 蒸馏。

        ``inputs`` 仍是 ``MyDataCollator`` 返回的可变长度 Python 列表。这里先
        恢复原文本，再分别生成学生/教师 tensor；因此数据集无需预先保存两套
        token ids。
        """

        input_ids = inputs['input_ids']
        labels = inputs['labels']

        # dataset.py 中 input_ids = prompt_ids + answer_ids，labels = answer_ids。
        # 例：len(input_id)=7、len(label)=3，则前 4 个 token 属于 prompt。
        prompt_ids = [input_id[:len(input_id) - len(label)] for input_id, label in zip(input_ids, labels)]

        # 先用学生 tokenizer 还原 JSON 文本。decode -> encode 不一定严格保持
        # 原 token 序列，但保证后续两种 tokenizer 接收相同的可见文本。
        prompt_texts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
        answer_texts = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 分别套用各自 chat template、编码并 padding。
        student_input_ids, student_labels, student_attention_mask = self.get_inputs_from_texts(self.tokenizer, prompt_texts, answer_texts)
        teacher_input_ids, teacher_labels, teacher_attention_mask = self.get_inputs_from_texts(self.teacher_tokenizer, prompt_texts, answer_texts)

        # 学生和教师理论上可以位于不同设备；不过 ULDLoss 最终需要在同一
        # 设备上同时运算两边 logits，当前多卡入口尚未显式处理跨设备搬运。
        student_input_ids = student_input_ids.to(self.model.device)
        student_labels = student_labels.to(self.model.device)
        student_attention_mask = student_attention_mask.to(self.model.device)

        teacher_input_ids = teacher_input_ids.to(self.teacher_model.device)
        teacher_labels = teacher_labels.to(self.teacher_model.device)
        teacher_attention_mask = teacher_attention_mask.to(self.teacher_model.device)
        
        # 学生前向保留计算图，梯度会通过 ULDLoss 回传到 LoRA 参数。
        student_outputs = model(input_ids=student_input_ids, attention_mask=student_attention_mask)


        self.teacher_model.eval()
        # 教师仅提供软目标，不保存激活梯度，可显著减少教师侧显存占用。
        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids=teacher_input_ids, attention_mask=teacher_attention_mask)

        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits

        # 学生/教师序列长度和词表维都可以不同，由 ULDLoss 内部完成两层对齐。
        loss = self.uld_loss_fn(student_logits, teacher_logits, student_labels, teacher_labels, student_input_ids, teacher_input_ids)

        return (loss, student_outputs) if return_outputs else loss
        

if __name__ == '__main__':
    # 主程序使用本地模型目录、example.json 和硬编码超参数，适合作为最小
    # 实验入口。正式训练时通常应将这些值改为命令行或配置文件参数。
    import os
    # DataLoader 已启用多个 worker；关闭 tokenizer 内部线程可避免并行冲突
    # 和 transformers 的 fork 警告。
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # 学生模型：规模较小，负责学习教师输出；路径相对于当前工作目录。
    model = AutoModelForCausalLM.from_pretrained("Qwen2.5-0.5B-Instruct",trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-0.5B-Instruct",trust_remote_code=True)

    # LoRA 只训练低秩 adapter。r 是秩，lora_alpha 控制 adapter 缩放，
    # target_modules 覆盖 attention 投影和 MLP 投影层。
    lora_config = LoraConfig(
    r=8,  
    lora_alpha=256,  
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.1, 
    task_type=TaskType.CAUSAL_LM)
 
    model = get_peft_model(model, lora_config)
    # 当前写法直接使用“当前 CUDA 设备”。使用 torchrun 时，每个进程应先
    # 根据 LOCAL_RANK 设置设备，否则多个进程可能先把模型加载到 GPU 0。
    model.cuda()
    # Trainer 日志通常会估算 FLOPs；此实验将其覆盖为 0，避免相关统计开销。
    model.floating_point_ops = lambda s: 0
    print(model.print_trainable_parameters())

    # 教师模型：只推理、不更新。当前未指定 torch_dtype，具体加载精度取决于
    # transformers/模型配置；9B 模型若以 FP32 加载会带来很高显存占用。
    teacher_tokenizer = AutoTokenizer.from_pretrained("glm-4-9b-chat",trust_remote_code=True)
    teacher_model = AutoModelForCausalLM.from_pretrained("glm-4-9b-chat",trust_remote_code=True)
    teacher_model.eval()
    teacher_model.cuda()
  
    
    # 训练参数示例：8 条 example 数据会在 batch_size=8 时组成一个 batch，
    # num_train_epochs=1 因而通常只执行一个 optimizer step。
    args = TrainingArguments(output_dir='./results', 
                            num_train_epochs=1, 
                            do_train=True, 
                            per_device_train_batch_size=8,
                            gradient_accumulation_steps=1,
                            logging_steps=1,
                            report_to='tensorboard',
                            save_strategy='steps',
                            save_total_limit=3,
                            save_steps=100,
                            bf16=True,
                            learning_rate=0.00001,
                            lr_scheduler_type='cosine',
                            dataloader_num_workers=8,
                            dataloader_pin_memory=True)
    # collator 保持可变长度列表；真正 padding 发生在 KGTrainer 内部。
    data_collator = MyDataCollator()
    train_dataset = SFTDataset('example.json', tokenizer=tokenizer)

    # tokenizer 参数表示学生 tokenizer；教师 tokenizer 通过自定义参数传入。
    trainer = KGTrainer(model=model,
                        teacher_model=teacher_model, 
                        args=args, 
                        train_dataset=train_dataset, 
                        tokenizer=tokenizer, 
                        teacher_tokenizer=teacher_tokenizer,
                        data_collator=data_collator)

    # 不恢复旧 checkpoint，从头训练；结束后分别保存模型/adapter 和 Trainer
    # 状态（优化器、scheduler、global step 等）。
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model('./saves')
    trainer.save_state()