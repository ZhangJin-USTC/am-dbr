# 方法与来源

## 带折扣的 PIT-CDF 跟踪器

本文用 Direct binned recalibration（DBR）表示 `framework/methods.py` 中实现的带折扣固定网格 PIT-CDF 跟踪器。它维护指数折扣的分箱质量，并用连续分段线性插值形成概率映射。带折扣的 PIT 更新、单调插值和在线预测聚合均建立在校准与预测文献的已有方法之上。

`DBR-Fixed` 是论文中的固定记忆比较方法。`DirectBinnedRecalibration` 另提供自适应折扣选项，用于补充实验。

## 自适应记忆聚合

`AM-DBR`（保存结果时的代码标签为 `AMDBR-core-noidentity`）组合七个 DBR 记忆尺度。它先输出一个凸组合 PIT-CDF，在观察到结果后用 PIT 尺度 CRPS 为各专家评分，再通过指数加权和固定份额更新权重。论文在给定假设下分析这一方法的切换记忆评分比较、分布跟踪界和覆盖率传递结果。

## 两阶段 rank/PAV 比较方法

`OCC-Fixed` 和 `OCC-Adaptive` 分别是补充材料中固定与自适应 rank/PAV 比较方法的结果标签。它们先估计折扣秩，再用加权保序回归拟合单调映射。这些方法与直接跟踪 CDF 的方法使用相同 PIT 序列，用于结构比较和状态量比较。

## 如何复现

方法注册、参数和按时间顺序的评价协议见 `docs/methods.md` 与 `docs/METHOD_IMPLEMENTATION_MAP.md`。参考 CSV 文件给出预期数值摘要。运行 `python reproduce.py all` 会执行各项研究，并将输出与参考摘要比较。数据哈希、源文件哈希和完整输出校验支持一致的复现，以及同一次运行的可靠断点续跑。
