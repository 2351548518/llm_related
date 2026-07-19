# ARPO 学习笔记

本目录整理的是 **Agentic Reinforced Policy Optimization（ARPO）**，重点解释
它如何在多轮工具调用中使用 entropy 自适应地创建 partial rollout，以及共享前缀
与分支 token 如何获得训练信号。

## 文件

- [ARPO_NOTE.md](./ARPO_NOTE.md)：中文主笔记，包含公式、流程图、数值例子和
  实现注意事项。
- [arpo_demo.py](./arpo_demo.py)：仅依赖 Python 标准库的可运行数值演示。
- [assets](./assets)：ARPO 自适应 rollout 总览图。

## 快速运行

从仓库根目录执行：

```bash
python arpo_note/arpo_demo.py
```

这份代码只模拟 entropy、分支预算和 advantage attribution，不会下载模型，也不会
真正调用搜索或 Python 工具。
