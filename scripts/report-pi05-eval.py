#!/usr/bin/env python3
"""Generate a local evaluation report without changing any raw results."""
import argparse
import json
from pathlib import Path

ORDER = ("place_a2b", "pick_color", "pick_spatial", "open_drawer",
         "stack_cubes", "joint_direction")


def load(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def generate(root, reference):
    current = load(root / "summary.json")
    distributed = load(root / "distributed_summary.json")
    previous = load(reference / "summary.json")
    rows = []
    for task in ORDER:
        now = current.get("task_results", {}).get(task)
        old = previous.get("task_results", {}).get(task)
        if now is None:
            continue
        rows.append({
            "task": task, "episodes": now["episode_count"],
            "execution_failures": now["failed_episode_count"],
            "score_60000": now["score"],
            "success_rate_60000": now["success_rate"],
            "score_30000": old["score"] if old else None,
            "success_rate_30000": old["success_rate"] if old else None,
            "score_delta": now["score"] - old["score"] if old else None,
            "success_delta_pp": (
                100 * (now["success_rate"] - old["success_rate"])
                if old else None
            ),
            "stage_success_rate": now.get("stage_success_rate", {}),
        })
    complete = (
        current.get("status") == "succeeded" and len(rows) == 6
        and all(row["episodes"] == 50 and row["execution_failures"] == 0
                for row in rows)
    )
    output = {
        "complete_300_episode_evaluation": complete,
        "run": str(root), "reference": str(reference),
        "status": current.get("status", "no_final_summary"),
        "score_60000": current.get("eval_average_score"),
        "success_rate_60000": current.get("eval_average_success_rate"),
        "score_30000": previous.get("eval_average_score"),
        "success_rate_30000": previous.get("eval_average_success_rate"),
        "elapsed_seconds": distributed.get("elapsed_seconds"),
        "tasks": rows,
    }
    (root / "comparison_60000_vs_30000.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )
    lines = [
        "# π₀.₅ 全量 SFT：60k 本地评测报告", "",
        "完整性：" + ("六任务 × 50 回合完整结束，无执行失败。" if complete
                     else "尚未取得完整、无执行失败的 300 回合结果；以下不能视为完整成绩。"),
        "", "本地评测结果，不代表平台正式成绩。`succeeded` 表示执行完成，不表示每个回合成功。",
        "", "## 配置", "",
        "- 权重：`/study/wangmengfan/hbot/checkpoints/pi05_hbot_full/60000`（含 params 与 assets）。",
        "- 模型：hbot-full，GPU 1，端口 8766，三路相机，每次 15 个动作。",
        "- 仿真：GPU 0，串行六任务，各 50 回合，seed=100000；每任务录制前 2 回合。",
        "- 未启用新双进程、CPU 线程/亲和性或性能分析选项，未修改功率上限。",
        "- 本轮配置与权重元数据摘要见 `run_manifest.json` 和 `config_snapshot/`。",
        "", "## 总体与逐任务成绩", "",
    ]
    if current:
        lines.append(
            f"60k：平均得分 {current['eval_average_score']:.4f}，"
            f"平均成功率 {current['eval_average_success_rate']:.2%}。"
        )
    if previous:
        lines.append(
            f"30k 对照：平均得分 {previous['eval_average_score']:.4f}，"
            f"平均成功率 {previous['eval_average_success_rate']:.2%}。"
        )
    if output["elapsed_seconds"] is not None:
        lines.append(f"本轮评测调度用时：{output['elapsed_seconds'] / 3600:.2f} 小时。")
    lines += ["", "| 任务 | 回合 | 60k 得分 | 60k 成功率 | 30k 得分 | 30k 成功率 | 分数变化 | 成功率变化 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        old_score = "—" if row["score_30000"] is None else f"{row['score_30000']:.4f}"
        old_rate = "—" if row["success_rate_30000"] is None else f"{row['success_rate_30000']:.2%}"
        delta = "—" if row["score_delta"] is None else f"{row['score_delta']:+.4f}"
        pp = "—" if row["success_delta_pp"] is None else f"{row['success_delta_pp']:+.2f} pp"
        lines.append(
            f"| {row['task']} | {row['episodes']} | {row['score_60000']:.4f} | "
            f"{row['success_rate_60000']:.2%} | {old_score} | {old_rate} | {delta} | {pp} |"
        )
    lines += ["", "## 困难阶段", "",
              "以下是各阶段的达成比例，不是相邻阶段的条件成功率。应结合视频区分抓取、放置、稳定性或指令理解问题。", ""]
    for row in rows:
        stages = "；".join(f"{name}={value:.1%}" for name, value in row["stage_success_rate"].items())
        lines.append(f"- {row['task']}：{stages or '无阶段统计'}；执行失败 {row['execution_failures']} 回合。")
    if complete:
        ranked = sorted(rows, key=lambda row: row["success_rate_60000"])
        lines += ["", "本次优先检查的低成功率任务：" + "、".join(
            f"{row['task']}（{row['success_rate_60000']:.1%}）" for row in ranked[:2]
        ) + "。"]
    lines += ["", "## 解读边界", "",
              "每任务 50 回合中，一次成功对应 2 个百分点。单次评测的涨跌不能单独证明训练步数增加必然改善泛化。",
              "沿用相同配置与主种子便于对照，但模型动作会改变轨迹，物理随机性与场景重采样仍可能造成差异。",
              "先区分执行失败与模型未完成任务，再结合阶段指标和前两个回合的视频分析。", "",
              "## 同目录产物", "",
              "- `summary.json`：榜单口径汇总；`distributed_summary.json`：执行与任务明细。",
              "- `comparison_60000_vs_30000.json`：可复核的对照数值。",
              "- 各任务 `shards/`：逐回合结果、日志、录制；`videos/`：导出视频。",
              "- `model_server.log`、`evaluation.log`：本次服务与评测日志。",
              "- `run_status.json`：任务最终状态，不应仅凭后台会话消失判断完成。", ""]
    (root / "report_60000_vs_30000.md").write_text("\n".join(lines), encoding="utf-8")
    return complete


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    print(json.dumps({"complete": generate(args.output, args.reference)}))
