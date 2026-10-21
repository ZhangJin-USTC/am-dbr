"""Create lightweight notebook backups for completed SOTA comparison runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nbformat as nbf


def make_notebook(dataset_dir: Path, title: str, output_path: Path) -> None:
    rel = str(dataset_dir)
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(f"# {title}\n\nThis notebook reads the saved SOTA comparison outputs for inspection."),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            f"root = Path(r'{rel}')\n"
            "summary = pd.read_csv(root / 'summary.csv')\n"
            "multi = pd.read_csv(root / 'multilevel_coverage.csv')\n"
            "records = pd.read_csv(root / 'records.csv')\n"
            "display(summary)\n"
            "display(multi)\n"
        ),
        nbf.v4.new_code_cell(
            "alpha = 0.10\n"
            "sub = records[records['alpha'].eq(alpha)].copy()\n"
            "fig, ax = plt.subplots(figsize=(7, 4.5))\n"
            "plot = (sub.groupby('method')\n"
            "          .agg(coverage=('covered','mean'), width=('width','mean'))\n"
            "          .reset_index())\n"
            "for _, row in plot.iterrows():\n"
            "    ax.scatter(row['width'], row['coverage'], s=70)\n"
            "    ax.text(row['width'], row['coverage'], '  ' + row['method'], va='center', fontsize=9)\n"
            "ax.axhline(1-alpha, color='black', linestyle='--', linewidth=1)\n"
            "ax.set_xlabel('Mean interval width')\n"
            "ax.set_ylabel('Coverage')\n"
            "ax.set_title('Coverage-width tradeoff')\n"
            "fig.tight_layout()\n"
        ),
        nbf.v4.new_code_cell(
            "alpha = 0.10\n"
            "window = min(400, max(50, records['time_index'].nunique() // 5))\n"
            "sub = records[records['alpha'].eq(alpha)].copy()\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "for method, g in sub.groupby('method'):\n"
            "    gg = g.sort_values('time_index')\n"
            "    roll = gg['covered'].rolling(window, min_periods=max(20, window//5)).mean()\n"
            "    ax.plot(gg['time_index'], roll, label=method, linewidth=2 if str(method).startswith('OCC') else 1.2)\n"
            "ax.axhline(1-alpha, color='black', linestyle='--', linewidth=1)\n"
            "ax.set_xlabel('Time index')\n"
            "ax.set_ylabel('Rolling coverage')\n"
            "ax.set_title(f'Rolling coverage, window={window}')\n"
            "ax.legend(ncol=2, fontsize=8)\n"
            "fig.tight_layout()\n"
        ),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    default_root = repo_root / "outputs" / "sota_occ"
    parser.add_argument("--root", default=str(default_root))
    parser.add_argument("--out-dir", default=str(default_root / "notebooks"))
    args = parser.parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "summary.csv").exists() and (child / "records.csv").exists():
            make_notebook(child, f"SOTA comparison: {child.name}", out_dir / f"SOTA_{child.name}.ipynb")
    synth = root / "synthetic"
    if synth.exists():
        for child in sorted(synth.iterdir()):
            if child.is_dir() and (child / "summary.csv").exists() and (child / "records.csv").exists():
                make_notebook(child, f"Synthetic SOTA comparison: {child.name}", out_dir / f"SOTA_synthetic_{child.name}.ipynb")


if __name__ == "__main__":
    main()
