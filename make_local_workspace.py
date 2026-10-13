"""Create editable IDE/notebook copies outside the verified code package.

Open this original file in an IDE and click Run without editing it. A sibling
workspace is created. This uses only the standard library and installs nothing.
"""
from pathlib import Path
import json
import sys

REPO = Path(__file__).resolve().parent


def create_workspace(destination=None, *, repo=None):
    code = Path(repo if repo is not None else REPO).expanduser().resolve()
    workspace = (code.parent / (code.name + "_local_verification") if destination is None
                 else Path(destination).expanduser().resolve())
    if workspace == code or workspace.is_relative_to(code):
        raise ValueError("The editable workspace must be outside the verified code package.")
    if not (code / "reproduce.py").is_file() or not (code / "local_verify.py").is_file():
        raise FileNotFoundError("The source directory is not an AM-DBR reproduction package.")
    names = ("run_from_IDE.py", "AM_DBR_local_verification.ipynb", "START_HERE.txt")
    conflicts = [name for name in names if (workspace / name).exists()]
    if conflicts:
        raise FileExistsError(f"Workspace already contains {conflicts}; open the existing copies or choose a new directory.")
    notebook = json.loads((code / "AM_DBR_local_verification.ipynb").read_text(encoding="utf-8"))
    notebook['cells'][0]['source'] = [
        '# AM-DBR v1.4.0 个人工作副本（科学核心 v1.2.0）\n',
        '\n本副本位于代码包之外，可以安全配置、运行与自动保存。CODE指向未修改的原始包，',
        'RUNS指向这个工作区的runs目录。请勿上传本副本：它含本机路径与运行后的输出。\n',
    ]
    cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    config = "".join(cells[0]["source"])
    config = config.replace("CODE = Path.cwd().resolve()  # Change if this notebook was opened elsewhere.",
                            f"CODE = Path({str(code)!r})  # Verified package; keep its files unchanged.")
    config = config.replace("RUNS = CODE / 'my_verification'", f"RUNS = Path({str(workspace / 'runs')!r})")
    cells[0]["source"] = config.splitlines(True)
    for cell in cells:
        cell["outputs"] = []
        cell["execution_count"] = None
    script = f'''"""Editable personal launcher; the verified package stays unchanged."""
from pathlib import Path
import sys

# Edit this copy, not files inside CODE. Point PYTHON at a locked Python 3.12 environment.
STEP = "check"  # install, check, smoke, data, all, resume, report, results
CODE = Path({str(code)!r})
PYTHON = Path(sys.executable)
RUNS = Path({str(workspace / 'runs')!r})
SNAPSHOT = None  # Optional Path to an authorized exact historical snapshot.
WORKERS = 1

sys.path.insert(0, str(CODE))
from local_verify import LocalVerification
verify = LocalVerification(CODE, PYTHON, RUNS, SNAPSHOT, WORKERS)
if __name__ == "__main__":
    for output_stream in (sys.stdout, sys.stderr):
        if hasattr(output_stream, "reconfigure"):
            output_stream.reconfigure(encoding="utf-8", errors="replace")
    verify.run(STEP)
'''
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / names[0]).write_text(script, encoding="utf-8")
    (workspace / names[1]).write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (workspace / names[2]).write_text(
        "This is your editable local workspace.\n"
        "Open run_from_IDE.py or AM_DBR_local_verification.ipynb here.\n"
        f"Read {code / 'LOCAL_VERIFICATION_zh.txt'} for environment and data setup.\n"
        "Keep the original code package unchanged, including its notebook template.\n"
        "Do not upload this workspace: it contains your local paths and generated results.\n",
        encoding="utf-8")
    print("Editable workspace:", workspace)
    print("Spyder / IDE:", workspace / names[0])
    print("Jupyter:", workspace / names[1])
    print("Keep the original package files unchanged. No software was installed.")
    return workspace


if __name__ == "__main__":
    create_workspace()
