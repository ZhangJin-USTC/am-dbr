# AM-DBR：自适应记忆直接分箱再校准

本匿名仓库对应论文 **Adaptive-Memory Online Distributional Recalibration under Distribution Shift**。

代码实现论文中的 AM-DBR 方法及比较方法。AM-DBR 在多个记忆尺度上维护折扣 PIT-CDF 跟踪器，并使用 fixed-share 正确评分聚合这些映射。Direct binned recalibration（DBR）作为固定记忆构件和比较方法保留。补充材料中的区间方法和 rank/PAV 结构比较实现也包含在仓库中。

## 快速开始

实际验证环境为 Windows x86-64、Python 3.12。运行数值研究不需要 GPU、R、MATLAB 或 LaTeX。

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
.venv\Scripts\python reproduce.py check
.venv\Scripts\python reproduce.py smoke
```

`check` 检查代码包和环境；`smoke` 运行一组完整但较短的模拟。要复现论文中的全部数值结果，运行：

```powershell
.venv\Scripts\python reproduce.py all --workers 1
```

完整研究需要较长 CPU 时间。程序会获取并核对所需的公开数据快照，运行模拟和真实数据研究，检查结果，并生成论文所需表格和图片。如果历史数据接口发生变化，请先用 `reproduce.py data --import-data 路径` 导入获得授权的精确快照，再重新运行。数据哈希不一致时程序会停止，不会静默使用不同数据。

已有完整结果时，可用 `reproduce.py report` 检查结果并重新生成论文资产；中断后可用 `reproduce.py resume` 从已核验检查点继续。不要让两个进程同时写同一个输出目录。

## 结果位置

- `outputs/paper_assets/generated`：表格、统计量、正文数字和图片；
- `outputs/main`：模拟和主要真实数据结果；
- `outputs/state`：匹配状态量敏感性结果；
- `outputs/windows`：报告窗口敏感性结果；
- `outputs/reference_comparison.json`：与参考摘要的比较；
- `outputs/RUN_STATUS.json`：整体运行状态。

完整复现应同时得到通过的 `RUN_STATUS.json`、参考结果比较和扩展资产检查记录。参考摘要用于核对实现，具备精确输入数据时仍应进行完整运行。

## 在 IDE 或 Jupyter 中运行

在 Spyder 或其他 Python IDE 中运行 `make_local_workspace.py`。它会在代码包外生成个人工作区和 Notebook 启动文件。请在生成的个人工作区中运行 Notebook 并保存输出；不要把个人环境、下载的数据或生成的结果上传到匿名仓库。

## 数据和许可证

仓库不重新分发第三方原始数据。数据来源、下载方法和引用要求见 `docs/data_access.md` 与 `docs/DATA_LICENSES_AND_ATTRIBUTION.md`。作者需要在解除匿名前为原创代码选择许可证；上游参考文件保留各自许可证声明。
