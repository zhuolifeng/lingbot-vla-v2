"""Create a reproducible episode split and optionally a checkpoint-matched config.

This script writes only small project files; dataset contents are never changed.
Run from the repository root.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import yaml


DATASETS = ('egg_overall_20260902', 'gift_overall_20260812', 'put_note_overall_20260910')


def checkpoint_horizon(checkpoint):
    config = json.loads((Path(checkpoint) / 'config.json').read_text())
    horizon = config.get('chunk_size', config.get('n_action_steps'))
    if horizon is None:
        # The official release currently only records vlm_family in config.json.
        # Match this repository's configuration default, rather than another VLA.
        if config.get('vlm_family') != 'qwen3_vl':
            raise ValueError('Unknown checkpoint without an action horizon')
        source = Path('lingbotvla/models/vla/lingbot_vla/configuration_lingbot_vla.py')
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.FunctionDef) and node.name == '__init__':
                arguments = node.args.args[-len(node.args.defaults):]
                for arg, default in zip(arguments, node.args.defaults):
                    if arg.arg == 'chunk_size':
                        horizon = ast.literal_eval(default)
        print(f'Checkpoint omits action horizon; using this repository configuration default: {horizon}.')
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError('Could not determine checkpoint action horizon')
    if 'n_action_steps' in config and config['n_action_steps'] != horizon:
        raise ValueError('Checkpoint chunk_size and n_action_steps disagree')
    return horizon


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=Path('/mnt/dataset'))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--tokenizer', type=Path)
    parser.add_argument('--depth-root', type=Path)
    parser.add_argument('--moge-checkpoint', type=Path)
    parser.add_argument('--video-root', type=Path)
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        parser.error('--val-fraction must be between zero and one')

    manifest = {'seed': args.seed, 'val_fraction': args.val_fraction,
                'trim_start': 30, 'trim_end': 30, 'datasets': {}}
    lines = []
    for name in DATASETS:
        root = args.data_root / name
        episodes = [json.loads(line) for line in (root / 'meta/episodes.jsonl').read_text().splitlines()]
        for ep in episodes:
            if ep['length'] <= 61 or not ep.get('task_annotation', '').strip():
                raise ValueError(f'Invalid length or missing English prompt: {name}/{ep["episode_index"]}')
        def order(ep):
            key = f'{args.seed}:{name}:{ep["episode_index"]}'
            return hashlib.sha256(key.encode()).digest()
        ordered = sorted(episodes, key=order)
        count = max(1, round(len(ordered) * args.val_fraction))
        val = sorted(ep['episode_index'] for ep in ordered[:count])
        train = sorted(ep['episode_index'] for ep in ordered[count:])
        manifest['datasets'][name] = {
            'root': str(root.absolute()), 'train': train, 'val': val,
            'episodes_sha256': hashlib.sha256((root / 'meta/episodes.jsonl').read_bytes()).hexdigest(),
            'retained_frames': sum(ep['length'] - 60 for ep in episodes),
            'sample_anchors': sum(ep['length'] - 61 for ep in episodes),
        }
        lines.append(f'umi {root.absolute()}\n')
        print(f'{name}: train={len(train)}, val={len(val)}, retained_frames={manifest["datasets"][name]["retained_frames"]}')

    output = Path('assets/training_data')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'umi_split.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'umi.txt').write_text(''.join(lines))

    if args.checkpoint:
        horizon = checkpoint_horizon(args.checkpoint)
        config = yaml.safe_load(Path('configs/vla/real_robot/real_robot.yaml').read_text())
        config['model']['model_path'] = str(args.checkpoint.absolute())
        if args.tokenizer:
            config['model']['tokenizer_path'] = str(args.tokenizer.absolute())
        config['data'].update(data_name='multi', train_path=str(output / 'umi.txt'),
                              episode_split='train', cameras=['camera_wrist_left', 'camera_wrist_right'],
                              num_workers=4)
        config['train'].update(chunk_size=horizon, output_dir='output/umi',
                               micro_batch_size=1, gradient_accumulation_steps=4,
                               global_batch_size=32, use_wandb=True,
                               wandb_project='lingbot-umi', wandb_name='umi-relative',
                               enable_gradient_checkpointing=True, save_steps=2000)
        depth = config['train']['align_params']['depth']
        video = config['train']['align_params']['video']
        if args.depth_root:
            depth['morgbd_path'] = str(args.depth_root.absolute() / 'model.pt')
        if args.moge_checkpoint:
            depth['moge_path'] = str(args.moge_checkpoint.absolute())
        if args.video_root:
            video['ckpt_path'] = str(args.video_root.absolute() / 'teacher_step_10000.pth')
            video['config_path'] = str(args.video_root.absolute() / 'config.yaml')
        target = Path('configs/vla/umi/umi.yaml')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(config, sort_keys=False))
        norm_config = {'data': {**config['data'], 'norm_path': 'assets/norm_stats/umi.json'},
                       'train': {'chunk_size': horizon, 'micro_batch_size': 32,
                                 'global_batch_size': 256, 'gradient_accumulation_steps': 1,
                                 'output_dir': 'output/umi_norm'}}
        (target.parent / 'norm.yaml').write_text(yaml.safe_dump(norm_config, sort_keys=False))
        print(f'Wrote {target} and norm.yaml; checkpoint action horizon={horizon}.')
        print('Fill any remaining /path/to teacher/tokenizer paths before training.')
    else:
        print('Split created. Pass --checkpoint to generate configs using its actual action horizon.')


if __name__ == '__main__':
    main()
