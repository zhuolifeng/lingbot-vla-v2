"""Read local LeRobot v2.1 episodes without invoking the v3 LeRobot loader.

Parquet rows and videos stay immutable. Trimming, episode splits, and action
padding are applied as a view, with timestamps in the original video timeline.
"""

import bisect
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from .video_utils import decode_video_frames


class LocalV21Metadata:
    def __init__(self, root):
        self.root = Path(root)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if self.info.get("codebase_version") != "v2.1":
            raise ValueError(f"Expected local LeRobot v2.1 data: {self.root}")
        self.fps = self.info["fps"]
        self.features = self.info["features"]
        self.episodes = {}
        for line in (self.root / "meta/episodes.jsonl").read_text().splitlines():
            episode = json.loads(line)
            idx = episode["episode_index"]
            if idx in self.episodes:
                raise ValueError(f"Duplicate episode {idx}: {self.root}")
            self.episodes[idx] = episode
        self.tasks = {
            item["task_index"]: item["task"]
            for item in map(json.loads, (self.root / "meta/tasks.jsonl").read_text().splitlines())
        }
        self.video_keys = [k for k, v in self.features.items() if v["dtype"] == "video"]
        self.camera_keys = self.video_keys

    def get_data_file_path(self, episode_index):
        return self.info["data_path"].format(
            episode_index=episode_index,
            episode_chunk=episode_index // self.info["chunks_size"],
        )

    def get_video_file_path(self, episode_index, video_key):
        return self.info["video_path"].format(
            episode_index=episode_index,
            episode_chunk=episode_index // self.info["chunks_size"],
            video_key=video_key,
        )


class LocalV21Dataset(Dataset):
    def __init__(self, metadata, *, delta_timestamps, image_keys, numeric_keys,
                 options, split="train", load_image=True, image_transforms=None,
                 video_backend="torchcodec"):
        self.meta = metadata
        self.root = metadata.root
        self.repo_id = str(self.root)
        self.load_image = load_image
        self.image_transforms = image_transforms
        self.video_backend = video_backend
        self.image_keys = list(image_keys)
        if not set(self.image_keys) <= set(metadata.video_keys):
            raise ValueError("Configured cameras are absent from the dataset")
        self.numeric_keys = sorted(set(numeric_keys) - set(metadata.video_keys))
        self.numeric_keys = sorted(set(self.numeric_keys) | {
            "timestamp", "frame_index", "episode_index", "index", "task_index",
        })
        self.delta_indices = {}
        for key, offsets in delta_timestamps.items():
            frames = np.asarray(offsets) * metadata.fps
            if not np.allclose(frames, np.rint(frames), atol=1e-4):
                raise ValueError(f"Offsets must align with the dataset FPS: {key}")
            self.delta_indices[key] = np.rint(frames).astype(np.int64)
        self.trim_start = int(options.get("trim_start", 0))
        self.trim_end = int(options.get("trim_end", 0))
        # UMI action[t] already stores state[t+1]. This only constrains the last
        # valid label; it never shifts or regenerates action values.
        self.action_target_offset = int(options.get("action_target_offset", 0))
        if min(self.trim_start, self.trim_end, self.action_target_offset) < 0:
            raise ValueError("Trim counts and action_target_offset must be nonnegative")
        self.prompt_field = options.get("prompt_field")
        selected = sorted(metadata.episodes)
        if options.get("split_file"):
            manifest = json.loads(Path(options["split_file"]).read_text())
            entry = manifest["datasets"][self.root.name]
            if Path(entry["root"]).resolve() != self.root.resolve():
                raise ValueError(f"Split manifest points to a different dataset: {self.root}")
            selected = entry[split]
            if len(selected) != len(set(selected)):
                raise ValueError("Split contains duplicate episodes")
            if set(entry["train"]) & set(entry["val"]):
                raise ValueError("Train and validation episodes overlap")
        self.episode_ids = []
        self.ends = []
        total = 0
        for idx in selected:
            episode = metadata.episodes[idx]
            length = episode["length"] - self.trim_start - self.trim_end - self.action_target_offset
            if length <= 0:
                raise ValueError(f"Episode {idx} is too short after trimming: {self.root}")
            if self.prompt_field and not str(episode.get(self.prompt_field, "")).strip():
                raise ValueError(f"Episode {idx} has no {self.prompt_field}")
            self.episode_ids.append(idx)
            total += length
            self.ends.append(total)
        if not total:
            raise ValueError(f"Empty split {split}: {self.root}")
        self._cache = OrderedDict()

    def __len__(self):
        return self.ends[-1]

    @property
    def num_frames(self):
        return len(self)

    @property
    def num_episodes(self):
        return len(self.episode_ids)

    def _episode_data(self, idx):
        if idx not in self._cache:
            table = pq.read_table(self.root / self.meta.get_data_file_path(idx), columns=self.numeric_keys)
            if len(table) != self.meta.episodes[idx]["length"]:
                raise ValueError(f"Episode length disagrees with metadata: {idx}")
            data = {}
            for key in self.numeric_keys:
                column = table[key].combine_chunks()
                values = column.to_numpy(zero_copy_only=False)
                if values.dtype == object:
                    values = np.stack(values)
                data[key] = torch.from_numpy(np.array(values, copy=True))
            frames = data["frame_index"].numpy()
            if not np.array_equal(frames, np.arange(len(table))):
                raise ValueError(f"Non-contiguous frame_index in episode {idx}")
            if not torch.all(data["episode_index"] == idx):
                raise ValueError(f"Incorrect episode_index in episode {idx}")
            timestamps = data["timestamp"].numpy()
            if not np.allclose(timestamps, frames / self.meta.fps, atol=1e-4):
                raise ValueError(f"Timestamps do not match video frame indices in episode {idx}")
            self._cache[idx] = data
            if len(self._cache) > 4:
                self._cache.popitem(last=False)
        self._cache.move_to_end(idx)
        return self._cache[idx]

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        pos = bisect.bisect_right(self.ends, index)
        episode_id = self.episode_ids[pos]
        frame = self.trim_start + index - (self.ends[pos - 1] if pos else 0)
        episode = self.meta.episodes[episode_id]
        end = episode["length"] - self.trim_end  # exclusive original-frame boundary
        data = self._episode_data(episode_id)
        item = {key: value[frame].clone() for key, value in data.items()}
        for key, offsets in self.delta_indices.items():
            if key in self.meta.video_keys:
                continue
            indices = frame + offsets
            stop = end - (self.action_target_offset if key.startswith("action") else 0)
            item[key + "_is_pad"] = torch.from_numpy((indices < self.trim_start) | (indices >= stop))
            indices = np.clip(indices, self.trim_start, stop - 1)
            item[key] = data[key][torch.from_numpy(indices)].clone()
        if self.load_image:
            for key in self.image_keys:
                indices = np.clip(frame + self.delta_indices.get(key, np.array([0])), self.trim_start, end - 1)
                timestamps = data["timestamp"][torch.from_numpy(indices)].tolist()
                images = decode_video_frames(
                    self.root / self.meta.get_video_file_path(episode_id, key),
                    timestamps, tolerance_s=1e-4, backend=self.video_backend,
                ).squeeze(0)
                item[key] = self.image_transforms(images) if self.image_transforms else images
        item["task"] = (episode[self.prompt_field].strip() if self.prompt_field
                        else self.meta.tasks[int(item["task_index"])])
        return item
