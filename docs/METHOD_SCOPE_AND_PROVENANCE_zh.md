# 方法来源与贡献边界（v1.2.0）

这份说明随匿名复现包公开，目的是避免把代码标签误读成优先权声明。

## 继承的基础构件

Direct binned recalibration（DBR）是带折扣的固定网格 PIT 映射构造，本文把
它作为组件和基线使用。PIT 映射更新、遗忘递推、单调插值、在线 CDF 聚合等
都是校准和预测文献中已有的思想。因此，本项目不声称创造了 DBR 这个名称、
折扣 PIT 质量、CRPS 聚合或 fixed-share 加权。

代码中的 `DBR-Fixed` 和 `DBR-Adaptive` 用于复现实验比较，对应
`framework/methods.py` 的 `DirectBinnedRecalibration`。

## 本文方法

`AM-DBR`（代码标签 `AMDBR-core-noidentity`）才是论文主方法：建立多个 DBR
记忆尺度组成的有限专家库，输出一个凸组合的 PIT-CDF，以 PIT 尺度 CRPS 更新
权重，再使用 fixed-share 转移。论文的贡献边界是这一特定组合及其在给定假设
下的切换记忆 oracle、不确定性跟踪和覆盖率传递分析；不把 DBR、CRPS 聚合或
fixed-share 单独包装成原创。

v1.2.0 不改数值算法和公共 API；呈现层生成器把 rank/PAV 从主表移到补充材料。
其中一个生成器属于锁定研究源清单，因此 `PUBLIC_SOURCE_MANIFEST.json` 如实
记录了新源指纹。旧 checkpoint 不可直接 resume，必须新建输出目录，不能改写
历史凭据来绕过验证。

## rank/PAV 的地位

`OCC-Fixed` 和 `OCC-Adaptive` 保留为历史结构比较器。它们估计 rank/PAV 校准
映射，可用于诊断和匹配状态量敏感性；它们不是 AM-DBR 的贡献，也不是运行
主方法所必需。完整定义和结果放在补充材料，正文只将它们作为简短的结构消融/比较。

## 复现结果应如何解读

随包提供的 CSV/NPZ 是继承的、已经核验的证据。`python reproduce.py all` 成功
表示包完整性、冻结协议、数据快照和数值摘要通过检查；它不构成优先权声明，
也不证明真实依赖序列一定满足理论假设。发行版本是 v1.2.0，内部证据协议
schema 标识 `1.0.0-rc.1` 保留；这不表示与旧版本源哈希兼容。
