# GSPO 学习笔记

本目录是一份结合公式、推导、数值例子和代码的 GSPO（Group Sequence
Policy Optimization）学习材料。

## 文件说明

- [GSPO_NOTE.md](./GSPO_NOTE.md)：主笔记，从 GRPO 推导到 GSPO。
- [gspo_demo.py](./gspo_demo.py)：仅依赖 Python 标准库的数值演示。
- [gspo_loss.py](./gspo_loss.py)：可放进训练代码的 PyTorch 核心损失实现。

原始截图中的目标函数、长度归一化、裁剪粒度和 MoE Routing Replay 等内容，
均已重新整理为正文中的公式、表格和文字说明，不需要依赖图片阅读。

## 快速运行

数值演示不需要安装第三方依赖：

```bash
python gspo_note/gspo_demo.py
```

PyTorch 损失文件只做核心算法展示，需要在安装了 PyTorch 的训练环境中导入：

```python
from gspo_note.gspo_loss import gspo_loss, group_relative_advantages
```

建议先读主笔记中的“一个完整的数值例子”，再对照两个代码文件。
