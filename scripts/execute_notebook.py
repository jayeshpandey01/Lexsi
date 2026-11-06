"""Executor script for notebooks/verification.ipynb.

Executes all 14 code cells in sequential order within a shared globals namespace,
capturing real stdout stream outputs, executing queries against the live Turso Cloud
database, updating the visual figures in results/visualizations/, and serializing the
live execution outputs back into notebooks/verification.ipynb.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path

# Ensure writable matplotlib config directory
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

NB_PATH = ROOT / "notebooks" / "verification.ipynb"
VIS_DIR = ROOT / "results" / "visualizations"
VIS_DIR.mkdir(parents=True, exist_ok=True)


def execute_notebook() -> None:
    print(f"Loading notebook: {NB_PATH}")
    with open(NB_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    shared_globals: dict = {
        "__file__": str(NB_PATH),
        "__name__": "__main__",
    }

    code_cells = [(idx, cell) for idx, cell in enumerate(nb["cells"]) if cell["cell_type"] == "code"]
    total_code = len(code_cells)
    print(f"Found {total_code} code cells to execute.\n")

    start_all = time.perf_counter()

    for seq, (cell_idx, cell) in enumerate(code_cells, start=1):
        code_str = "".join(cell.get("source", []))
        first_line = code_str.strip().split("\n")[0] if code_str.strip() else "EMPTY"
        print(f"[{seq}/{total_code}] Executing Cell #{cell_idx} ({first_line[:50]}...)...")

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        cell_start = time.perf_counter()
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            try:
                exec(code_str, shared_globals)
            except Exception as e:
                print(f"\nERROR in Cell #{cell_idx}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                raise e
        cell_elapsed = (time.perf_counter() - cell_start) * 1000.0

        output_text = stdout_buf.getvalue()
        stderr_text = stderr_buf.getvalue()
        if stderr_text:
            output_text += ("\nSTDERR:\n" + stderr_text)

        # Update cell outputs: preserve any display_data image blocks, update stdout stream
        new_outputs = []
        if output_text:
            new_outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": [line + "\n" for line in output_text.splitlines()],
            })

        for old_out in cell.get("outputs", []):
            if old_out.get("output_type") == "display_data":
                new_outputs.append(old_out)

        cell["outputs"] = new_outputs
        cell["execution_count"] = seq
        print(f"       -> Done in {cell_elapsed:.1f}ms. Output lines: {len(output_text.splitlines())}")

    total_elapsed = time.perf_counter() - start_all
    print(f"\nAll {total_code} cells executed successfully in {total_elapsed:.2f}s!")

    # Write back updated notebook
    with open(NB_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Saved updated notebook with live execution outputs to {NB_PATH}")

    # Verify visual charts
    print("\nVisualizations in results/visualizations/:")
    for p in sorted(VIS_DIR.glob("*.png")):
        stat = p.stat()
        print(f"  • {p.name} ({stat.st_size:,} bytes, mtime: {time.ctime(stat.st_mtime)})")


if __name__ == "__main__":
    execute_notebook()

