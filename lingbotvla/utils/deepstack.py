"""DeepStack feature addition with static-shape, functional tensor operations."""

import torch


def add_deepstack_features(hidden_states, visual_pos_masks, visual_embeds):
    """Add packed visual features at mask positions without nonzero or mutation.

    Visual rows follow the flattened mask's True positions in row-major order,
    as in Qwen3-VL. Row zero is a sentinel for non-visual positions, including
    the case with no visual tokens. All index tensors have fixed output shape.
    """
    mask = visual_pos_masks.to(device=hidden_states.device, dtype=torch.bool).reshape(-1)
    visual_embeds = visual_embeds.to(device=hidden_states.device, dtype=hidden_states.dtype)
    packed = torch.cat((visual_embeds.new_zeros((1, visual_embeds.shape[-1])), visual_embeds), dim=0)
    indices = torch.where(mask, mask.to(torch.int64).cumsum(dim=0), 0)
    updates = packed.index_select(0, indices).reshape_as(hidden_states)
    return hidden_states + updates
