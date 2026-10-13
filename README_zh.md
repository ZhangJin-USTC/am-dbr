# AM-DBR: Adaptive-Memory Direct Binned Recalibration

中文名称：自适应记忆直接分箱再校准。对应论文 **v1.4.0 / R18**；上传包装标识 **R18-repository-20261013**。仓库标识和发布状态见 `REPOSITORY_IDENTITY.json`。

说明文档包：**docs-v1.4.0**；数值研究核心：**code-v1.2.0**，展示与报告层为 v1.4.0。本次未改变数值算法、参数、数据快照或参考指标 CSV。详见 `DOCS_RELEASE.json`。

不想用系统控制台：在Spyder打开 `make_local_workspace.py`，不修改原文件，
直接点击Run。它在代码包之外生成 `run_from_IDE.py` 与Notebook个人工作副本，
再编辑/运行这些外部副本。不要编辑包内local_verify.py或直接运行/保存包内
Notebook模板，否则完整性哈希会失败。Jupyter自动保存只应作用于外部副本。
逐步说明见 `LOCAL_VERIFICATION_zh.txt`。个人工作区含本机路径，不应公开上传。
本版数值方法、基线、参数、数据与参考CSV继承rc.5；同时把论文中的方法边界
写清楚。Direct binned recalibration（DBR）是已有的 PIT 映射校准构件和基线，
不是本文声称的原创基本算子。本文主方法是 AM-DBR：在多个 DBR 记忆尺度上建立
有限专家库，用 fixed-share 聚合，并分析切换记忆、跟踪和覆盖率传递。`OCC-Fixed`
与 `OCC-Adaptive` 的 rank/PAV 实现仍保留用于历史结构比较/消融，不能写成本文的
主贡献。v1.4.0 没有改动数值算法或 API；表格生成器把 rank/PAV 移到补充材料。
本次保留 code-v1.2.0 的全部 24 个锁定源文件和原有源指纹；新增报告生成器位于该锁定集合之外。源指纹相同的 v1.2.0 完整输出经过来源、配置和内容核验后可以重新生成报告。更早的 rc.5/v1.1.0 输出使用不同指纹，不能直接 resume 或传给普通 report；本地历史数组只有经过独立兼容性审计后才可使用单独的历史证据流程。本次没有重新拟合或重调数值方法。方法来源和 claim 边界见
`docs/METHOD_SCOPE_AND_PROVENANCE.md`（中文说明见
`docs/METHOD_SCOPE_AND_PROVENANCE_zh.md`）。

本包包括当前文章的方法和全部基线。在新目录解压，完整替换旧版复现入口；
不要混用不同版本的程序。安装 Python 3.12 后，在本文件所在目录运行：

Windows PowerShell：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python reproduce.py all --workers 1
```

macOS / Linux：

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python reproduce.py all --workers 1
```

最后一条命令检查环境、测试与上游基线，取得并核对五份数据，执行正文模拟、
真实数据、匹配状态量和报告起点敏感性，最后验证保存数组并生成全部数值
表格与图，还会逐项比较论文参考指标。默认单进程降低并行内存要求。

**需要与论文一致的历史数据快照。** 远程接口可能改变历史响应。当前访问
情况见 [数据状态](docs/data_access.md)。哈希不匹配就停止，不会用变化后的
数据冒充论文复现。取得作者有权提供的快照后运行：

```text
python reproduce.py data --import-data 快照文件夹或压缩包
python reproduce.py all
```

此处 `python` 应替换成上面的虚拟环境 Python 路径。代码公开不能消除
第三方历史数据访问限制。完整本地交付的快照与匿名仓库代码包分开保存，
原始数据是否可以公开须遵守来源许可。

中断后运行 `python reproduce.py resume`；先快速检查可用 `check`，
先跑一组完整模拟可用 `smoke`。`smoke` 包含全部 15 方法、5 水平，但
不等于整篇文章复现。`report` 对现有完整输出再次验证和生成图表。
`--out 新目录` 可以保存独立复现，`--workers 2` 可以增加并行单元数。
不要同时启动两个程序写同一目录。完整研究是长时间 CPU 计算，不是几分钟
演示。无需 GPU、R、MATLAB 或 LaTeX 来计算结果并生成图表。

| 输出位置 | 内容 |
|---|---|
| `outputs/paper_assets/generated` | 正文与补充全部数值表格、统计、叙述数字、PDF/PNG 图 |
| `outputs/main` | 48 模拟 + 5 真实数据单元，每个 15 方法、5 水平 |
| `outputs/state` | 48 匹配状态量单元，每个 2 个额外设置 |
| `outputs/windows` | 3 报告起点 × 5 数据集，每个 15 方法 |
| `outputs/reference_comparison.json` | 全部指标和论文参考值比较 |
| `outputs/environment.json`、`commands.log` | 环境与运行日志 |

比较容差为相对 `1e-6`、绝对 `1e-8`，排除运行时间。不能承诺任何电脑、
操作系统和未来数据接口都无条件获得逐字节相同的结果。Windows x86-64 / 
Python 3.12.x 是实际验证环境。已提供三平台 CI 与 Dockerfile；提供
配置不代表已经在三平台和 Docker 上分别运行整篇论文。

完成单元有来源、配置、身份和内容哈希凭据。源文件或结果被修改会拒绝复用。
主研究和报告起点研究保留未提交的方法文件到 `_interrupted` 后，从已核验
的方法继续；状态量研究把未完成单元保存到
`state/_interrupted` 后重新计算。已有完成单元损坏不会被默默覆盖。

部分数值证据保留内部协议 schema 标识 `1.0.0-rc.1`；该标识不决定 checkpoint 兼容性。兼容性仍由实际源指纹、参数和内容核验决定。算法、参数、数据、参考数值及 code-v1.2.0 的 24 源文件组合指纹均不变；论文、解释和本次报告包装为 v1.4.0。
不要把协议标识误认为古老算法。方法限定见 [methods.md](docs/methods.md)。
参考 CSV 是当前修正研究的已审查结果，不是新的盲测，也不能代替重新运行。
项目原始代码的许可证仍需作者确定，见 `LICENSE_STATUS.md`。

完整复现成功应同时看到 `RUN_STATUS.json` 中 all/resume/report 命令为 passed，
完整 `reference_comparison.json` 为 passed，以及 `paper_assets/generated/expanded_validation.json` 中 passed 为 true、cells 为 53、method_runs 为 795。`LocalVerification.show_results()` 会同时检查这些记录；仅 check/smoke 成功不够。

扩展报告由 `revision_tools/build_expanded_comparisons.py` 生成，普通 `reproduce.py report` 会自动调用它。它生成 4 张正文结果表、11 张情景/数据集详细表、1 张配对种子区间评分表和 2 张覆盖率—区间宽度图，并保留全部 15 个方法的指标 CSV。新增 90% 区间评分是对已锁定预测的补充评价，没有用它重新选择参数。各文件及 LaTeX 标签的对应关系见 `docs/artifact_map.md`。
