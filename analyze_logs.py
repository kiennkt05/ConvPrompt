"""Compare ConvPrompt and RainbowPrompt training logs.

This utility parses the textual logs produced by the original ConvPrompt
training pipeline (e.g. `conv_log.txt`) and the new RainbowPrompt pipeline
(`rainbow_log.txt`).  It extracts:

* Final summary metrics (average accuracy, forgetting, etc.)
* Per-task evaluation accuracy
* Total training time when available

Usage::

    python analyze_logs.py --conv conv_log.txt --rainbow rainbow_log.txt

The script prints a human-readable comparison table.  Use ``--json`` to obtain
the structured results in JSON format instead.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


AVG_SEEN_LINE = re.compile(
    r"\[Average accuracy till task(?P<task>\d+)\]\s+Acc@1: (?P<acc1>[\d.]+)"
    r"\s+Acc@5: (?P<acc5>[\d.]+)\s+Loss: (?P<loss>[\d.]+)"
    r"(?:\s+Forgetting: (?P<forgetting>-?[\d.]+))?"
    r"(?:\s+Backward: (?P<backward>-?[\d.]+))?"
)

SUMMARY_LINE_RAINBOW = re.compile(
    r"\[Rainbow avg till task(?P<tasks>\d+)\]\s+Acc@1: (?P<acc1>[\d.]+)"
    r"\s+Overall: (?P<overall>[\d.]+)\s+Forgetting: (?P<forgetting>[\d.]+)"
)

TASK_HEADER_CONV = re.compile(r"Test: \[Task (?P<task>\d+)\]")
TASK_RESULT_CONV = re.compile(
    r"\* Acc@1 (?P<acc1>[\d.]+) Acc@5 (?P<acc5>[\d.]+) loss (?P<loss>[\d.]+)"
)

TASK_RESULT_RAINBOW = re.compile(
    r"Rainbow Eval Task\[(?P<task>\d+)/(?P<total>\d+)\].*Acc:\s*(?P<acc>[\d.]+)"
)

TOTAL_TIME_CONV = re.compile(r"Total training time:\s*(?P<hours>\d+):(\d+):(\d+)")


def _parse_conv_log(path: Path) -> Dict[str, object]:
    per_task: Dict[int, Dict[str, float]] = {}
    summary: Dict[str, float] = {}
    seen_stats: List[Dict[str, float]] = []
    total_time: Optional[str] = None
    current_task: Optional[int] = None

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            header_match = TASK_HEADER_CONV.search(line)
            if header_match:
                current_task = int(header_match.group("task"))
                continue

            result_match = TASK_RESULT_CONV.search(line)
            if result_match and current_task is not None:
                per_task[current_task] = {
                    "acc1": float(result_match.group("acc1")),
                    "acc5": float(result_match.group("acc5")),
                    "loss": float(result_match.group("loss")),
                }
                continue

            avg_seen = AVG_SEEN_LINE.search(line)
            if avg_seen:
                entry = {
                    "task": int(avg_seen.group("task")),
                    "acc1": float(avg_seen.group("acc1")),
                    "acc5": float(avg_seen.group("acc5")),
                    "loss": float(avg_seen.group("loss")),
                }
                if avg_seen.group("forgetting") is not None:
                    entry["forgetting"] = float(avg_seen.group("forgetting"))
                if avg_seen.group("backward") is not None:
                    entry["backward"] = float(avg_seen.group("backward"))
                seen_stats.append(entry)
                summary = {
                    "tasks": entry["task"],
                    "acc1": entry["acc1"],
                    "acc5": entry["acc5"],
                    "loss": entry["loss"],
                    "forgetting": entry.get("forgetting"),
                    "backward": entry.get("backward"),
                }
                continue

            total_time_match = TOTAL_TIME_CONV.search(line)
            if total_time_match:
                total_time = total_time_match.group(0).split(": ", 1)[1]

    return {
        "per_task": per_task,
        "summary": summary,
        "total_time": total_time,
        "seen_curve": seen_stats,
    }


def _parse_rainbow_log(path: Path) -> Dict[str, object]:
    per_task: Dict[int, Dict[str, float]] = {}
    summary: Dict[str, float] = {}
    seen_stats: List[Dict[str, float]] = []

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            task_match = TASK_RESULT_RAINBOW.search(line)
            if task_match:
                task_idx = int(task_match.group("task"))
                per_task[task_idx] = {"acc1": float(task_match.group("acc"))}
                continue

            avg_seen = AVG_SEEN_LINE.search(line)
            if avg_seen:
                entry = {
                    "task": int(avg_seen.group("task")),
                    "acc1": float(avg_seen.group("acc1")),
                    "acc5": float(avg_seen.group("acc5")),
                    "loss": float(avg_seen.group("loss")),
                }
                if avg_seen.group("forgetting") is not None:
                    entry["forgetting"] = float(avg_seen.group("forgetting"))
                if avg_seen.group("backward") is not None:
                    entry["backward"] = float(avg_seen.group("backward"))
                seen_stats.append(entry)
                summary = {
                    "tasks": entry["task"],
                    "acc1": entry["acc1"],
                    "acc5": entry["acc5"],
                    "loss": entry["loss"],
                    "forgetting": entry.get("forgetting"),
                    "backward": entry.get("backward"),
                }
                continue

            summary_match = SUMMARY_LINE_RAINBOW.search(line)
            if summary_match:
                summary = {
                    "tasks": int(summary_match.group("tasks")),
                    "acc1": float(summary_match.group("acc1")),
                    "overall": float(summary_match.group("overall")),
                    "forgetting": float(summary_match.group("forgetting")),
                }

    return {
        "per_task": per_task,
        "summary": summary,
        "total_time": None,  # Rainbow logs currently do not emit total runtime
        "seen_curve": seen_stats,
    }


def _format_table(headers: List[str], rows: List[List[str]]) -> str:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, value in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(value))

    def fmt_row(row: List[str]) -> str:
        return " | ".join(value.ljust(col_widths[idx]) for idx, value in enumerate(row))

    separator = "-+-".join("-" * w for w in col_widths)
    lines = [fmt_row(headers), separator]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def compare_logs(conv_path: Path, rainbow_path: Path) -> Dict[str, object]:
    conv = _parse_conv_log(conv_path)
    rainbow = _parse_rainbow_log(rainbow_path)

    return {
        "conv": conv,
        "rainbow": rainbow,
    }


def _build_human_report(results: Dict[str, object]) -> str:
    conv = results["conv"]
    rainbow = results["rainbow"]

    lines: List[str] = []
    lines.append("## Summary Metrics")

    headers = ["Metric", "ConvPrompt", "RainbowPrompt"]
    rows: List[List[str]] = []

    conv_summary = conv.get("summary", {})
    rainbow_summary = rainbow.get("summary", {})

    def fmt_value(summary: Dict[str, float], key: str) -> str:
        value = summary.get(key)
        return f"{value:.4f}" if isinstance(value, float) else "N/A"

    rows.append(["Average Acc@1", fmt_value(conv_summary, "acc1"), fmt_value(rainbow_summary, "acc1")])
    rows.append(["Average Acc@5", fmt_value(conv_summary, "acc5"), "N/A"])
    rows.append(["Overall Acc@1", "N/A", fmt_value(rainbow_summary, "overall")])
    rows.append(["Average Loss", fmt_value(conv_summary, "loss"), "N/A"])
    rows.append(["Forgetting", fmt_value(conv_summary, "forgetting"), fmt_value(rainbow_summary, "forgetting")])
    rows.append(["Backward Transfer", fmt_value(conv_summary, "backward"), "N/A"])

    conv_time = conv.get("total_time") or "N/A"
    rainbow_time = rainbow.get("total_time") or "N/A"
    rows.append(["Total Training Time", conv_time, rainbow_time])

    lines.append(_format_table(headers, rows))
    lines.append("")

    lines.append("## Per-Task Acc@1 (%)")
    task_rows: List[List[str]] = []
    max_task = max(
        max(conv["per_task"].keys(), default=0),
        max(rainbow["per_task"].keys(), default=0),
    )
    for task_idx in range(1, max_task + 1):
        conv_val = conv["per_task"].get(task_idx, {}).get("acc1")
        rainbow_val = rainbow["per_task"].get(task_idx, {}).get("acc1")
        task_rows.append([
            f"Task {task_idx}",
            f"{conv_val:.2f}" if conv_val is not None else "N/A",
            f"{rainbow_val:.2f}" if rainbow_val is not None else "N/A",
        ])

    lines.append(_format_table(headers, task_rows))
    lines.append("")

    lines.append("## Average Accuracy / Forgetting Over Seen Classes")
    seen_rows: List[List[str]] = []
    max_seen = max(
        len(conv.get("seen_curve", [])),
        len(rainbow.get("seen_curve", [])),
    )
    for idx in range(max_seen):
        def row_value(data: Dict[str, object], index: int, key: str) -> str:
            curve = data.get("seen_curve", [])
            if index < len(curve) and key in curve[index]:
                return f"{curve[index][key]:.4f}"
            return "N/A"

        task_label = f"Task {idx + 1}"
        conv_acc = row_value(conv, idx, "acc1")
        conv_forget = row_value(conv, idx, "forgetting")
        rainbow_acc = row_value(rainbow, idx, "acc1")
        rainbow_forget = row_value(rainbow, idx, "forgetting")

        seen_rows.append([
            task_label,
            f"Acc@1 {conv_acc} / For. {conv_forget}",
            f"Acc@1 {rainbow_acc} / For. {rainbow_forget}",
        ])

    lines.append(_format_table(headers, seen_rows))

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ConvPrompt vs RainbowPrompt logs")
    parser.add_argument("--conv", required=True, type=Path, help="Path to conv_log.txt")
    parser.add_argument("--rainbow", required=True, type=Path, help="Path to rainbow_log.txt")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text")
    args = parser.parse_args()

    results = compare_logs(args.conv, args.rainbow)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(_build_human_report(results))


if __name__ == "__main__":
    main()

