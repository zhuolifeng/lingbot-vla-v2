"""Fixed-subset, distributed action-loss validation during VLA training."""

import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Subset


class StrictValidationDataset(Dataset):
    """Do not let MultiVLADataset replace failed validation samples at random."""
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if hasattr(self.dataset, 'getdata'):
            return self.dataset.getdata(index)
        return self.dataset[index]


def fixed_validation_indices(dataset_size, sample_count, seed, world_size, batch_size):
    if sample_count <= 0 or sample_count > dataset_size:
        raise ValueError('eval_samples must be positive and no larger than the validation dataset')
    if sample_count % (world_size * batch_size):
        raise ValueError('eval_samples must be divisible by world_size * micro_batch_size')
    # Sorting preserves the randomly selected subset while improving episode/video locality.
    return sorted(random.Random(seed).sample(range(dataset_size), sample_count))


def build_validation_loader(dataset, indices, collate_fn, batch_size, rank, world_size,
                            seed, num_workers=0, pin_memory=True):
    subset = Subset(StrictValidationDataset(dataset), indices)
    sampler = DistributedSampler(subset, num_replicas=world_size, rank=rank,
                                 shuffle=False, drop_last=False)
    return DataLoader(
        subset, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=False,
        generator=torch.Generator().manual_seed(seed + rank),
        persistent_workers=num_workers > 0,
    )


@contextmanager
def validation_context(model, device, seed):
    """Preserve training RNG streams and each module's train/eval mode."""
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] \
        if device.type == 'cuda' else []
    modes = [(module, module.training) for module in model.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            model.eval()
            with torch.no_grad():
                yield
    finally:
        for module, training in modes:
            module.training = training
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def evaluate_action_loss(model, dataloader, device, seed):
    """Evaluate masked flow-matching loss, weighted by valid action elements.

    Uses the same noise/time distribution as training with a repeatable RNG
    stream. This is a validation loss, not rollout success or denoised pose error.
    Teacher/auxiliary losses are skipped by the model's action_loss_only path.
    All FSDP ranks must call this function with equally sized loader shards.
    """
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    with validation_context(model, device, seed):
        for batch in dataloader:
            batch = {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                     for key, value in batch.items()
                     if key not in ('rep_id', 'pil_images', 'future_pil_images',
                                    'future_video_effective_fps')}
            outputs = model(**batch, action_loss_only=True)
            loss = outputs[1].detach().to(torch.float64)
            valid = batch['joint_mask'].sum().to(torch.float64)
            totals[0] += loss * valid
            totals[1] += valid
            totals[2] += batch['actions'].shape[0]
    if dist.is_initialized():
        dist.all_reduce(totals)
    if totals[1].item() <= 0 or not torch.isfinite(totals).all():
        raise ValueError('Validation produced non-finite loss or no valid action elements')
    return {'vla_loss': (totals[0] / totals[1]).item(),
            'samples': int(totals[2].item())}
