# scripts/gcg_pretty_dump.py
from pathlib import Path
import json

def _rows(jsonl_path: Path):
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except Exception:
            continue

def pretty_dump(log_dir: str):
    """
    Convert all *.jsonl under log_dir to *.txt (human-readable).
    Usage from a Python shell:
        from scripts.gcg_pretty_dump import pretty_dump
        pretty_dump("gcg_logs/gcg/len20_100step_bs64_seed0_l50_t1.0_static_k128")
    """
    root = Path(log_dir)
    for jsonl in root.rglob("*.jsonl"):
        txt = jsonl.with_suffix(".txt")
        lines = [
            "# GCG human-readable log (converted)",
            f"path={jsonl}",
            "",
            "step/num_steps | loss | best_loss | begin_with | in_response | queries | time_min | suffix | generated_head",
            "-" * 88,
            "",
        ]
        for r in _rows(jsonl):
            gen = r.get("generated", "")
            if isinstance(gen, str) and len(gen) > 160:
                gen = gen[:160] + "…"
            lines.append(
                f"{r.get('step')}/{r.get('num_steps', 'NA')} | "
                f"{r.get('loss')} | {r.get('best_loss')} | "
                f"{int(r.get('success_begin_with', False))} | "
                f"{int(r.get('success_in_response', False))} | "
                f"{r.get('queries')} | "
                f"{float(r.get('time_min', 0.0)):.2f} | "
                f"{r.get('suffix')} | {gen}"
            )
        txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("Wrote", txt)

# If you insist on one-liner use without imports:
if __name__ == "__main__":
    # Default to converting the repo's default log root if present
    pretty_dump("gcg_logs")
