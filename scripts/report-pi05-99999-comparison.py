#!/usr/bin/env python3
"""Audit full local evaluations and compare them without changing raw results."""
import argparse
import copy
import json
import math
from pathlib import Path

ORDER = ("place_a2b", "pick_color", "pick_spatial", "open_drawer",
         "stack_cubes", "joint_direction")
REFERENCES = (
    ("LoRA 40k", "local_pi05_lora_39999_20261028_140639_740492_10276"),
    ("LoRA 60k", "local_pi05_lora_59999_20261029_202633_3586617_17288"),
    ("Full SFT 30k", "local_pi05_full_30000_20261030_170853_3664253_7601"),
    ("Full SFT 60k", "local_pi05_full_60000_20261031_141137"),
    ("Full SFT 80k (2 Clients)", "local_pi05_full_80000_parallel_20261101_083416"),
)


def load(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def summarize(root, label):
    summary = load(root / "summary.json")
    rows = {}
    problems = []
    for task in ORDER:
        episodes = []
        for path in sorted((root / task / "shards").glob("*/eval_result.json")):
            episodes.extend(load(path).get("episode_results", []))
        n = len(episodes)
        final = summary.get("task_results", {}).get(task, {})
        scores = [float(item["progress"]) for item in episodes]
        successes = sum(bool(item["success"]) for item in episodes)
        score = sum(scores) / n if n else None
        rate = successes / n if n else None
        valid = (
            n == 50 and all(math.isfinite(s) and 0 <= s <= 1 for s in scores)
            and all(item.get("stop_reason") in ("success", "max_steps")
                    for item in episodes)
            and final.get("episode_count") == 50
            and final.get("failed_episode_count") == 0
            and math.isclose(final.get("score", -1), score or 0, abs_tol=1e-7)
            and math.isclose(final.get("success_rate", -1), rate or 0, abs_tol=1e-7)
        )
        if not valid:
            problems.append(task + ": incomplete or inconsistent results")
        rows[task] = {
            "episodes": n, "successes": successes, "score": score,
            "success_rate": rate, "validated": valid,
            "stage_success_rate": final.get("stage_success_rate", {}),
        }
    complete = summary.get("status") == "succeeded" and not problems
    mean_score = sum(r["score"] for r in rows.values()) / 6 if complete else None
    mean_rate = sum(r["success_rate"] for r in rows.values()) / 6 if complete else None
    if complete and (not math.isclose(summary.get("eval_average_score", -1), mean_score, abs_tol=1e-7)
                     or not math.isclose(summary.get("eval_average_success_rate", -1), mean_rate, abs_tol=1e-7)):
        complete = False
        problems.append("overall summary differs from episode results")
        mean_score = mean_rate = None
    return {
        "label": label, "path": str(root), "complete": complete,
        "episodes": sum(r["episodes"] for r in rows.values()),
        "successes": sum(r["successes"] for r in rows.values()),
        "mean_score": mean_score, "mean_success_rate": mean_rate,
        "tasks": rows, "problems": problems,
    }


def prepare_previews(root, count=2):
    """Limit only preview metadata; preserve all scores and raw episode files."""
    source = root / "summary.json"
    summary = load(source)
    if not isinstance(summary.get("task_results"), dict):
        raise ValueError("Missing leaderboard summary")
    backup = root / "summary_before_preview_limit.json"
    if not backup.exists():
        with backup.open("x", encoding="utf-8") as stream:
            stream.write(source.read_text())
    result = copy.deepcopy(summary)
    for task in result["task_results"].values():
        task["preview_episodes"] = [
            item for item in task.get("preview_episodes", [])
            if 0 <= item["episode_index"] < count
        ]
    temporary = root / "summary.preview.tmp"
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(source)
    return sum(len(t["preview_episodes"]) for t in result["task_results"].values())


def generate(root):
    current = summarize(root, "Full SFT 99999 (3 Clients)")
    references = [summarize(root.parent / name, label) for label, name in REFERENCES]
    distributed = load(root / "distributed_summary.json")
    payload = {"current": current, "references": references,
               "elapsed_seconds": distributed.get("elapsed_seconds"),
               "execution": distributed.get("execution"),
               "comparison_caveat": "99999 uses 3 simulation workers, 80k used 2 and older runs were serial. Model RNG and fallback scene sampling can differ."}
    (root / "comparison_all_models.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# π₀.₅ Full SFT 99999（约 100k）三并发评测对比", "",
             "完整性：" + ("六任务各 50 回合，300 回合逐条复核通过。" if current["complete"] else "结果尚不完整或校验未通过，不能作为完整成绩。"),
             "", "本地结果，非平台正式成绩。", "",
             "## 配置与对照边界", "",
             "- 权重：`/study/wangmengfan/hbot/checkpoints/pi05_hbot_full/99999`。",
             "- GPU 1：单模型服务，三相机，15 步动作，保持原模型采样设置。",
             "- GPU 0：三个独立仿真进程，每进程 8 线程，CPU 亲和性 0-23,48-71。",
             "- 六任务各 50 回合；叠方块分 4 片，物体放置分 2 片；每任务只录制前 2 回合。",
             "- 主种子与任务预算不变。并行改变 Client 会话/RNG 序列，场景重采样也可能不同；99999 为 3 并发，80k 为 2 并发，更早结果为串行，不是严格逐轨迹对照。",
             "- 未修改显卡功率、物理频率、相机频率或最大步数。", "",
             "## 总体与分项结果", "",
             "每项：得分 / 成功率。", "",
             "| 模型 | 平均 | Place_A2B | Pick_Color | Pick_Spatial | Open_Drawer | Stack_Cubes | Joint_Direction |",
             "|---|---|---|---|---|---|---|---|"]
    for row in [current, *reversed(references)]:
        if not row["complete"]:
            lines.append(f"| {row['label']} | 未完成/未通过校验 | — | — | — | — | — | — |")
            continue
        cells = [f"{row['tasks'][task]['score']:.3f} / {row['tasks'][task]['success_rate']:.2%}" for task in ORDER]
        lines.append(f"| {row['label']} | {row['mean_score']:.4f} / {row['mean_success_rate']:.2%} | " + " | ".join(cells) + " |")
    if payload["elapsed_seconds"] is not None:
        lines += ["", f"仿真调度用时：{payload['elapsed_seconds']/3600:.2f} 小时（不含后续视频导出）。"]
    lines += ["", "## 99999 阶段指标", ""]
    for task, row in current["tasks"].items():
        stages = "；".join(f"{name}={rate:.1%}" for name, rate in row["stage_success_rate"].items())
        lines.append(f"- {task}：{row['successes']}/{row['episodes']}；{stages or '无最终阶段统计'}。")
    lines += ["", "## 产物", "",
              "`run_status.json` 区分仿真完整性与视频导出状态；`resource_samples.jsonl` 记录资源。",
              "`comparison_all_models.json` 保存可复核数值；`config_snapshot/` 保存配置；各任务 `shards/` 保存逐回合结果。", ""]
    (root / "report_99999_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return current["complete"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps({"complete": generate(args.output)}))
