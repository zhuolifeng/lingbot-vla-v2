"""Batched local-v2.1 normalization statistics on a torch device.

Pass one computes stable moments and global bounds. Pass two uses fixed global
histogram bins, so quantiles do not depend on rank partitioning or rebinning.
Only one episode of raw data and one batch of action chunks occupy the device.
"""

import torch
import torch.distributed as dist
from tqdm import tqdm

from .local_v21_dataset import LocalV21Dataset
from ...utils.normalize import NormStats


class TorchStats:
    def __init__(self, dim, device, bins=5000):
        self.count = 0
        self.mean = torch.zeros(dim, device=device, dtype=torch.float64)
        self.m2 = torch.zeros_like(self.mean)
        self.low = torch.full_like(self.mean, float('inf'))
        self.high = torch.full_like(self.mean, -float('inf'))
        self.hist = torch.zeros((dim, bins), device=device, dtype=torch.int64)
        self.bins = bins

    def update(self, values):
        if values.shape[-1] != self.mean.numel():
            raise ValueError('Feature dimensions differ between normalization datasets')
        values = values.reshape(-1, self.mean.numel()).to(torch.float64)
        n = values.shape[0]
        if not n:
            return
        if not torch.isfinite(values).all():
            raise ValueError('Non-finite data encountered while computing normalization')
        variance, mean = torch.var_mean(values, dim=0, correction=0)
        total = self.count + n
        delta = mean - self.mean
        self.m2 += variance * n + delta.square() * (self.count * n / total)
        self.mean += delta * (n / total)
        self.low = torch.minimum(self.low, values.amin(dim=0))
        self.high = torch.maximum(self.high, values.amax(dim=0))
        self.count = total

    def synchronize_moments(self):
        if dist.is_initialized():
            count = torch.tensor(self.count, device=self.mean.device, dtype=torch.int64)
            dist.all_reduce(count)
            total = int(count.item())
            weighted_mean = self.mean * self.count
            dist.all_reduce(weighted_mean)
            global_mean = weighted_mean / max(total, 1)
            self.m2 += self.count * (self.mean - global_mean).square()
            dist.all_reduce(self.m2)
            dist.all_reduce(self.low, op=dist.ReduceOp.MIN)
            dist.all_reduce(self.high, op=dist.ReduceOp.MAX)
            self.mean = global_mean
            self.count = total
        if self.count < 2:
            raise ValueError('At least two valid values are required for normalization')

    def update_histogram(self, values):
        if values.shape[-1] != self.mean.numel():
            raise ValueError('Feature dimensions differ between normalization datasets')
        values = values.reshape(-1, self.mean.numel()).to(torch.float64)
        width = self.high - self.low
        width = torch.where(width > 0, width, torch.ones_like(width))
        indices = ((values - self.low) / width * self.bins).floor().long()
        indices.clamp_(0, self.bins - 1)
        offsets = torch.arange(self.mean.numel(), device=values.device) * self.bins
        counts = torch.bincount((indices + offsets).flatten(), minlength=self.hist.numel())
        self.hist += counts.reshape_as(self.hist)

    def finish(self):
        if dist.is_initialized():
            dist.all_reduce(self.hist)
        if not torch.all(self.hist.sum(dim=1) == self.count):
            raise ValueError('Histogram count disagrees with moment count')
        cumulative = self.hist.cumsum(dim=1).contiguous()
        quantiles = {}
        for name, fraction in [('q01', .01), ('q99', .99), ('q02', .02), ('q98', .98)]:
            targets = torch.full((self.mean.numel(), 1), fraction * self.count,
                                 device=self.mean.device, dtype=torch.float64)
            indices = torch.searchsorted(cumulative.to(torch.float64), targets).squeeze(1)
            quantiles[name] = self.low + (self.high - self.low) * indices / self.bins
        values = dict(mean=self.mean, std=(self.m2 / self.count).clamp_min(0).sqrt(),
                      min=self.low, max=self.high, **quantiles)
        return NormStats(**{key: value.cpu().numpy() for key, value in values.items()})


def validate_local_datasets(datasets):
    for wrapped in datasets:
        reader, transform = wrapped.dataset, wrapped.feature_transform
        if not isinstance(reader, LocalV21Dataset):
            raise ValueError('CUDA norm currently requires reader: local_v21')
        if transform.actions_convert_from_state:
            raise ValueError('CUDA norm requires explicit action columns, not actions derived from state')
        if transform.normalizer is not None or not transform.return_item_befor_padding:
            raise ValueError('CUDA norm requires unnormalized features before padding')
        if wrapped.transform is not None:
            raise ValueError('CUDA norm does not support additional dataset transforms')
        if any(key in reader.delta_indices for key in transform.org_features['states']):
            raise ValueError('CUDA norm requires a single current observation state')
        offsets = [reader.delta_indices[key] for key in transform.org_features['actions']]
        if not offsets or any(x.shape != offsets[0].shape or not (x == offsets[0]).all()
                              for x in offsets):
            raise ValueError('CUDA norm requires matching action timestamps')


@torch.inference_mode()
def iter_local_batches(datasets, device, batch_size, rank=0, world_size=1,
                       progress=False, description='GPU norm'):
    """Match LocalV21Dataset + FeatureTransform, vectorized across sample starts."""
    jobs = [(wrapped, ep) for wrapped in datasets for ep in wrapped.dataset.episode_ids]
    for wrapped, episode_id in tqdm(jobs[rank::world_size], desc=description,
                                    unit='episode', disable=not progress):
        reader, transform = wrapped.dataset, wrapped.feature_transform
        columns = set(transform.org_features['states'] + transform.org_features['actions'])
        raw_cpu = reader._episode_data(episode_id)
        raw = {key: raw_cpu[key].to(device) for key in columns}
        end = reader.meta.episodes[episode_id]['length'] - reader.trim_end
        stop = end - reader.action_target_offset
        action_offsets = {key: torch.as_tensor(reader.delta_indices[key], device=device)
                          for key in transform.org_features['actions']}
        for start in range(reader.trim_start, stop, batch_size):
            frames = torch.arange(start, min(start + batch_size, stop), device=device)
            # A singleton horizon dimension broadcasts each reference state
            # across its complete action chunk in the existing pose transform.
            item = {key: raw[key][frames].unsqueeze(1)
                    for key in transform.org_features['states']}
            for key, offsets in action_offsets.items():
                indices = frames[:, None] + offsets[None, :]
                item[key + '_is_pad'] = (indices < reader.trim_start) | (indices >= stop)
                item[key] = raw[key][indices.clamp(reader.trim_start, stop - 1)]
            yield transform.apply(item)
        del raw


@torch.inference_mode()
def compute_local_norm(dataset, device, batch_size=2048, rank=0, world_size=1,
                       progress=True, bins=5000):
    if batch_size < 1:
        raise ValueError('norm_batch_size must be positive')
    datasets = dataset._datasets
    validate_local_datasets(datasets)
    states = datasets[0].state_features
    actions = datasets[0].action_features
    # One CPU sample provides dimensions even on a rank with no assigned episode.
    sample = datasets[0][0]
    stats = {key: TorchStats(sample[key].shape[-1], device, bins) for key in states + actions}

    def values(batch, key):
        value = batch[key]
        if key in actions:
            value = value[~batch['action_is_pad']]
        return value.reshape(-1, value.shape[-1])

    for batch in iter_local_batches(datasets, device, batch_size, rank, world_size,
                                    progress and rank == 0, 'Norm moments (1/2)'):
        for key, accumulator in stats.items():
            accumulator.update(values(batch, key))
    for accumulator in stats.values():
        accumulator.synchronize_moments()

    for batch in iter_local_batches(datasets, device, batch_size, rank, world_size,
                                    progress and rank == 0, 'Norm quantiles (2/2)'):
        for key, accumulator in stats.items():
            accumulator.update_histogram(values(batch, key))
    result = {key: accumulator.finish() for key, accumulator in stats.items()}
    return result, stats[states[0]].count
