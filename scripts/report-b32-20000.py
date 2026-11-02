"""Audit the batch32 20k evaluation and compare all six previous models."""
import importlib.util
import json
from pathlib import Path

ROOT = Path('/study/wangmengfan/hbot')
REFERENCES = (
    ('LoRA 40k', 'local_pi05_lora_39999_20261028_140639_740492_10276'),
    ('LoRA 60k', 'local_pi05_lora_59999_20261029_202633_3586617_17288'),
    ('Full SFT 30k', 'local_pi05_full_30000_20261030_170853_3664253_7601'),
    ('Full SFT 60k', 'local_pi05_full_60000_20261031_141137'),
    ('Full SFT 80k (2 Clients)', 'local_pi05_full_80000_20261101_083416'),
    ('Full SFT 99999 (3+2 mixed)', 'local_pi05_full_99999_20261102'),
)


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate(root):
    helper = load_module(ROOT / 'scripts/report-pi05-99999-comparison.py', 'history_audit')
    strict = load_module(ROOT / 'scripts/report-pi05-99999-supplement.py', 'strict_audit')
    current = helper.summarize(root, 'Full SFT batch32 20k (1+2 Clients)')
    audits = {t: strict.audit_task(root, t) for t in helper.ORDER}
    current['complete'] = current['complete'] and all(r['complete'] for r in audits.values())
    refs = [helper.summarize(root.parent / name, label) for label, name in REFERENCES]
    payload = dict(current=current, references=refs, raw_audits=audits,
        comparison_caveat='Different training batch/step and simulator concurrency; not a controlled single-variable experiment.')
    (root / 'comparison_all_models.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    lines = ['# π₀.₅ batch32 · 20000 步本地评测', '',
        '完整性：' + ('六任务各50回合、共300回合逐条复核通过。' if current['complete'] else '尚未完整完成或完整性校验未通过。'), '',
        '- 权重：`/study/wangmengfan/hbot/checkpoints/pi05_hbot_32b/20000`。',
        '- GPU0：模型推理 + 1 个仿真 Client；GPU1：2 个仿真 Client；合计3并发。',
        '- 六任务各50回合，种子100000；Stack_Cubes四分片、Place_A2B两分片；每任务前2回合录制。',
        '- 三相机、15步动作、原去噪配置；每Client 8线程，CPU亲和性0-23,48-71。',
        '- GPU功率、任务步数和物理/渲染频率不变。',
        '- 独立预热会话不计入评测，也不推进评测会话的模型RNG。',
        '- 不同训练batch/步数、Client会话RNG与场景重采样均可能影响对比；不是严格单变量实验。', '',
        '每格为得分 / 成功率；本地结果不是平台正式成绩。', '',
        '| 模型 | 平均 | Place_A2B | Pick_Color | Pick_Spatial | Open_Drawer | Stack_Cubes | Joint_Direction |',
        '|---|---|---|---|---|---|---|---|']
    for row in [*refs, current]:
        if not row['complete']:
            lines.append(f"| {row['label']} | 未完成（{row['episodes']}/300） | — | — | — | — | — | — |")
        else:
            cells = [f"{row['tasks'][t]['score']:.3f} / {row['tasks'][t]['success_rate']:.2%}" for t in helper.ORDER]
            lines.append(f"| {row['label']} | {row['mean_score']:.4f} / {row['mean_success_rate']:.2%} | " + ' | '.join(cells) + ' |')
    lines += ['', '## 本轮任务进度与阶段指标', '']
    for task, row in audits.items():
        lines.append(f"- {task}: {row['successes']}/{row['episodes']}；阶段命中：{row['stage_counts']}。")
    lines += ['', '配置、日志、原始成绩、视频、资源采样与本报告保存在本目录。', '']
    (root / 'report_b32_20000_comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    return current['complete']
