v1.2.0 本地验证：Spyder / Jupyter / Python IDE
============================================================

适用范围
这份说明的“安装”和“运行”都可以在 IDE 的 Python 控制台或 Notebook 单元格中
执行，不要求使用 Windows 命令提示符或 PowerShell。程序子进程的输出会显示
在同一个 IDE / Notebook 内。默认一次只运行一个实验单元，减少内存占用。
需要 Python 3.12.x。无需 GPU、R、MATLAB 或 LaTeX 来计算结果和生成图表。
v1.2.0 保留 rc.5 的数值方法、基线、数据快照和参考值，新增便携 IDE 入口，
并明确 DBR 是已有基础构件，AM-DBR 是本文的特定主方法；rank/PAV 仅为历史
结构比较/消融。方法边界详见 `docs/METHOD_SCOPE_AND_PROVENANCE_zh.md`。

1. 推荐软件与目录
Spyder 路线：安装 Spyder standalone（独立安装器）和 Python 3.12.x。
Notebook 路线：使用已有 JupyterLab / Jupyter Notebook，或从 Anaconda Navigator
图形界面创建 Python 3.12 环境并启动 JupyterLab。不要改动 Anaconda base 环境。
从 ZIP 解压得到一个新目录，目录内应直接包含 reproduce.py、local_verify.py、
RELEASE.json、requirements-lock.txt、make_local_workspace.py。
包内local_verify.py与Notebook也是完整性清单保护文件：不要编辑、直接运行/保存
包内Notebook。先打开make_local_workspace.py并点击Run（不要修改它），会在代码
目录旁生成 <代码目录名>_local_verification；以下称WORKSPACE。里面的
run_from_IDE.py、AM_DBR_local_verification.ipynb才是可编辑工作副本。该步骤不安装软件。
如果已有工作副本，直接打开它；生成器不会覆盖已有个人配置。
目录可以含空格和中文。以下示例路径仅为占位，替换为自己解压的位置。

官方安装与解释器说明（2026-09-20 核对）：
https://www.python.org/downloads/
https://www.spyder-ide.org/
https://docs.spyder-ide.org/current/faq.html
https://jupyter.org/install

2. 先准备独立的计算环境（一次即可）
Spyder 自带解释器的版本可能不是 3.12。推荐把图形界面环境与论文计算环境分开。
在 Spyder 的 IPython Console 或 Notebook 单元格执行下面的 Python 代码。
把 CODE 改成代码目录，把 PYTHON312 改成已经安装的 Python 3.12 可执行文件。
Windows 通常名为 python.exe；macOS/Linux 通常为 python3.12。

    from pathlib import Path
    import subprocess, os, sys
    CODE = Path(r"C:/research/AM_DBR_v1.4.0")
    PYTHON312 = Path(r"C:/path/to/Python312/python.exe")
    subprocess.run([str(PYTHON312), "-c",
                    "import sys; assert sys.version_info[:2] == (3, 12), sys.version"], check=True)
    WORKSPACE = CODE.parent / (CODE.name + "_local_verification")
    ENV = WORKSPACE / ".venv"
    subprocess.run([str(PYTHON312), "-m", "venv", str(ENV)], check=True)
    PYTHON = ENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    subprocess.run([str(PYTHON), "-m", "pip", "install", "-r",
                    str(CODE / "requirements-lock.txt")], check=True)

这一步不需要激活环境、不需要改变 PowerShell 执行策略。安装过程需联网。
只应使用论文随包的 requirements-lock.txt，不要手工升级其中的版本。
如果已有一个独立的 Python 3.12 计算环境，也可以直接把 PYTHON 指向它。

如希望 Spyder 直接使用这个解释器：在 Tools / Preferences / Python interpreter
选择 .venv/Scripts/python.exe（Windows）或 .venv/bin/python，然后重启 kernel。
如 Spyder 提示缺少 spyder-kernels，按其显示的兼容范围安装；不同 Spyder 版本
所需范围不同，不要硬猜。可在原来可用的 IDE 中执行：
    subprocess.run([str(PYTHON), "-m", "pip", "install",
                    "spyder-kernels<按Spyder提示的上限,>=按提示的下限"], check=True)
上面最后一个字符串必须用 Spyder 实际给出的包要求替换，不是可原样运行的命令。
使用下面的子进程入口时，不必把 Spyder kernel 改成计算环境。

如希望在 Jupyter 中选择论文专用 kernel，可在原有可用 Notebook 中执行：
    subprocess.run([str(PYTHON), "-m", "pip", "install", "ipykernel"], check=True)
    subprocess.run([str(PYTHON), "-m", "ipykernel", "install", "--user",
                    "--name", "am-dbr-v140", "--display-name", "AM-DBR v1.4.0 (Python 3.12)"], check=True)
然后在 Notebook 的 Kernel 菜单选择 AM-DBR v1.4.0 (Python 3.12)。
若当前 kernel 已是论文专用 Python 3.12，可使用：
    %pip install -r "C:/research/AM_DBR_v1.4.0/requirements-lock.txt"
%pip 只适用于 IPython/Notebook 单元格，不应粘贴进普通 .py 文件；安装后重启 kernel。
Jupyter/spyder-kernels 是界面工具，未纳入论文数值依赖锁；安装后仍需运行 check。

3. Spyder 的最直接用法
打开WORKSPACE中的run_from_IDE.py。只在这个外部副本中把PYTHON改为计算环境路径。
不要改CODE指向目录中的local_verify.py，也不要修改其他受manifest保护的文件。
默认 STEP="check"，点击绿色 Run。以后只需改 STEP 并再次 Run。
如果本地交付提供了获授权的完整历史数据 ZIP，把 SNAPSHOT 改成 Path(该ZIP路径)。
没有快照时保留 None，data 步骤将尝试下载并核验来源快照。
RUNS决定输出目录，外部工作副本默认使用WORKSPACE/runs。不要与旧版输出混用。
代码包内的my_verification只是库默认值；本指南采用外部工作副本的runs路径。

4. Notebook 的用法
打开WORKSPACE中的AM_DBR_local_verification.ipynb，按顺序运行单元格。生成器已把
CODE指向未修改的原代码包。不要从原代码包直接打开/运行/保存Notebook。
在外部工作副本的配置单元格设置CODE、
PYTHON、SNAPSHOT；Notebook 调用的是与 Spyder 相同的 LocalVerification 类。
公开包内Notebook模板不含作者机器路径、保存输出或执行结果；个人工作副本含
本机路径且执行后会保存输出，不应上传匿名仓库。运行Jupyter的自动保存只改变
外部副本，因此不会破坏原包哈希。
“安装”与“完整运行”单元默认被注释，避免点击 Run All 后意外开始长计算；
首次运行按说明取消相应一行前的 #。

5. 建议验证顺序及成功标志
check → smoke → data → all → results。
check：核对 Python/依赖版本/包文件哈希和24个数值源指纹，运行全部测试，
       再对 FACI、SF-OGD、SAOCP、NExCP 的固定上游递推做20组/6200次预测比较。
       正常结束时 RUN_STATUS.json 为 passed；check 不生成全部论文结果。
smoke：一个1500观测模拟单元，包含全部15种方法和5个水平，对比参考指标。
       WORKSPACE/runs/smoke_check/smoke_reference_comparison.json 必须 passed。
       smoke 无需真实数据，但它不是完整复现。
data：导入授权快照或下载5份数据，并逐一核对哈希。
      GUI入口写到 WORKSPACE/runs/data_snapshot；记录在 full_run/data_validation.json。
all：使用已准备并通过校验的数据；执行48个模拟+5个真实数据主研究单元、
     48个状态量敏感性单元和15个报告起点单元。最后重算表图并验收全部参考指标。
     此步骤是长时间CPU工作。请保持电脑唤醒，不要同时启动两个程序写同一目录。
results：只读取验收凭据。只有完整运行状态和全部四项研究比较均通过时才显示
         FULL VERIFICATION PASSED。check/smoke/data 的 passed 不会被当成完整成功。

生成图表位于 WORKSPACE/runs/full_run/paper_assets/generated。
完整 reference_comparison.json 应有 synthetic=720、real=75、state=96、windows=225
行，每项16个数值指标。比较采用相对1e-6、绝对1e-8容差，不比较运行时间。
累计1116行验收不代表1116个独立样本，也不代表新的盲测实验。

6. 中断、继续与再次核验
中断后等待子进程与worker完全停止，把 STEP 改为 resume，保留同一个 RUNS。
不要为了通过检查而改哈希或删除完成凭据。resume 只接受来源/配置/内容匹配的
完成单元；未提交的中断文件会保留后重算。report 验证完整已有结果并重生成图表。
本版改变了正文表格生成器，因此组合源指纹已经升级。旧 rc.5/v1.1.0 的
checkpoint不可直接resume；不要改写它们的来源凭据。用全新的RUNS完整运行。
不必编辑包内源码来配置路径：外部脚本/Notebook导入原包的LocalVerification类，
该类将路径参数传给同一reproduce.py，原始文件哈希仍能严格核对。
v1.2.0 本次验证记录与继承的rc.3全研究结果明确分开，不声称本次重跑了所有拟合。

7. 常见问题
Use Python 3.12：当前 kernel 或 PYTHON 选错；打印 sys.executable、sys.version。
Missing/expected dependency：用 PYTHON -m pip 安装lock；Jupyter不要用不明来源的 !pip。
Package file missing or modified：可能上传漏文件/换行被Git转写/混入旧代码；重新解压
完整ZIP到新目录，保留.gitattributes。不要编辑冻结包内文件，也不要在包内保存
运行后的Notebook；用make_local_workspace.py生成的外部副本。
Data hash mismatch：上游数据已变化，需使用获授权的精确历史快照；不能绕过验证。
No module named local_verify：CODE必须是包含local_verify.py的那一级目录。
Full verification is not complete：查看 full_run/commands.log 与 RUN_STATUS.json；
先确认已运行 all 或 resume，并完成四项研究；smoke成功不够。
Reference comparison failed：保留日志，检查依赖/硬件差异和数组；不要手改参考CSV。
内存不足：保持WORKERS=1；关闭其他大程序；完整研究仍可能需要足够内存和磁盘。
文件锁或旧worker：先停止旧任务，再resume；不要并行写同一RUNS。

本说明基于Windows/Python 3.12的实际检查和短测；Linux/macOS/Docker配置存在，
不等于已在这些平台执行完整论文研究。独立运行的数值结果验收以本机凭据为准。
