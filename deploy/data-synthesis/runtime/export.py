"""Convert accepted MCAPs and verify actions, video alignment and 15-step windows."""
import argparse
import io
import json
from pathlib import Path

import av
import numpy as np
from PIL import Image
from mcap.reader import make_reader
from mcap_protobuf.decoder import DecoderFactory

import conversion
from audit_raw import CAMS
from common import require, contained, sha, save


def pixel_check(source, output, index, offset, n):
    results = []
    for camera, topic in CAMS.items():
        frames = []
        with source.open('rb') as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for _, _, _, msg in reader.iter_decoded_messages(topics=[topic]):
                im = Image.open(io.BytesIO(msg.data)).convert('RGB').resize((256,144))
                frames.append(np.asarray(im, dtype=np.int16))
        videos = list(output.glob(f'videos/*/observation.images.{camera}/episode_{index:06d}.mp4'))
        require(len(videos) == 1, 'Missing or duplicate camera video')
        errors = []
        with av.open(str(videos[0])) as video:
            for j, frame in enumerate(video.decode(video=0)):
                rgb = Image.fromarray(frame.to_ndarray(format='rgb24')).resize((256,144))
                errors.append(float(np.abs(np.asarray(rgb, dtype=np.int16) - frames[offset+j]).mean()))
        require(len(errors) == n and max(errors) < 6, 'Source/video frame alignment failed')
        results.append(dict(camera=camera, max_pixel_mae=max(errors), frames=n))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--name', required=True)
    args = p.parse_args()
    root = Path('/work')
    state = json.loads((root/'status.json').read_text())
    require(state['status'] != 'running', 'Pause collection before exporting a stable snapshot')
    rows = json.loads((root/'manifest.json').read_text())
    require(bool(rows), 'No accepted samples')
    base = root/'exports'/args.name
    base.mkdir(parents=True, exist_ok=False)
    entries = []
    for row in rows:
        source = contained(root, row['mcap_relative'])
        require(sha(source) == row['mcap_sha256'], 'MCAP changed')
        require(row['raw_gate_pass'] and row['dedup_gate_pass'], 'Unqualified manifest row')
        entries.append(dict(mcap_path=str(source), seed=row['seed'],
                            instruction=row['metadata']['instruction']))
    conversion.BASE, conversion.OUTPUT = base, base/'lerobot_v21'
    report = dict(status='converting', training_ready=False, source_count=len(rows))
    save(base/'report.json', report)
    try:
        conversion.convert_dataset(entries)
        output = conversion.OUTPUT
        info = json.loads((output/'meta/info.json').read_text())
        require(info['codebase_version'] == 'v2.1' and info['fps'] == 30 and
                info['total_episodes'] == len(rows), 'Wrong dataset format/count')
        ok, reason = conversion.verify_labels(entries)
        require(ok, reason)
        audits, masks = [], []
        for i, (entry, row) in enumerate(zip(entries, rows)):
            result = conversion.audit_episode(entry, i)
            require(result['pass'], f'Numeric/video integrity failed: episode {i}')
            result['pixel_alignment'] = pixel_check(Path(entry['mcap_path']), output,
                i, result['raw_offset'], result['frames'])
            # Keep original frames. Only sampler anchors are filtered.
            valid = [j-result['raw_offset'] for j in row['valid_15_step_anchor_indices']
                     if result['raw_offset'] <= j and
                     j-result['raw_offset']+15 <= result['frames']]
            require(len(valid) >= 30, 'Insufficient aligned 15-step training anchors')
            masks.append(dict(episode_index=i, valid_anchor_indices=valid,
                              horizon=15, source=row['mcap_relative']))
            audits.append(result)
            save(base/'episode_audits.json', audits)
            print(f'EXPORT_AUDIT {i+1}/{len(rows)}', flush=True)
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset('hbot/synthetic-v1', root=str(output),
            download_videos=False, video_backend='pyav',
            delta_timestamps={'action':[i/30 for i in range(15)]})
        start = 0
        for i, mask in enumerate(masks):
            anchors = mask['valid_anchor_indices']
            # LeRobot 0.3.3 may flag floating-point episode end as padding.
            # Exact final-window integrity was already verified numerically.
            for anchor in (anchors[0], anchors[len(anchors)//2]):
                sample = dataset[start+anchor]
                require(tuple(sample['action'].shape) == (15,8) and
                        tuple(sample['observation.state'].shape) == (8,) and
                        not sample['action_is_pad'].any(), 'Loader shape/padding mismatch')
            start += audits[i]['frames']
        require(len(dataset) == start, 'Reader total frame count mismatch')
        save(base/'training_anchor_masks.json', masks)
        report.update(status='format_verified', format_verified=True,
                      accepted_episodes=len(entries), training_ready=False,
                      note='Apply anchor masks. Team merge-check and visual review remain required.')
        save(base/'report.json', report)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        save(base/'report.json', report)
        raise


if __name__ == '__main__':
    main()
