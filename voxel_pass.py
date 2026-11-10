import numpy as np
from scipy import ndimage


def _world_to_grid(points, origin, voxel_size, resolution):
    """Convert world points to a clipped integer voxel index."""
    idx = np.floor((points - origin) / voxel_size).astype(np.int32)
    return np.clip(idx, 0, resolution - 1)


def _safe_row_normalize(weights, fallback=None):
    """Normalize rows to sum to 1, with a closest-source fallback for empty rows."""
    if weights.size == 0:
        return weights

    weights = np.asarray(weights, dtype=np.float32)
    row_sums = weights.sum(axis=1, keepdims=True)
    zero_rows = row_sums <= 1e-12

    if np.any(zero_rows):
        if fallback is not None:
            weights[zero_rows[:, 0]] = fallback[zero_rows[:, 0]].astype(np.float32)
            row_sums = weights.sum(axis=1, keepdims=True)
            row_sums[row_sums <= 1e-12] = 1.0
            weights = weights / row_sums
        else:
            weights[zero_rows[:, 0]] = 0.0
            row_sums = weights.sum(axis=1, keepdims=True)
            row_sums[row_sums <= 1e-12] = 1.0
            weights = weights / row_sums
    else:
        weights = weights / np.maximum(row_sums, 1e-12)

    return weights


def geodesic_voxel_transfer(
    source_verts: np.ndarray,
    source_weights: np.ndarray,
    target_verts: np.ndarray,
    resolution: int = 64,
    diffusion_steps: int = 8,
    falloff: float = 1.0,
    mix_factor: float = 1.0,
):
    """
    Lightweight volumetric weight transfer.

    This takes source vertices weighted by bone groups, seeds those weights into a
    regular grid, diffuses them through the volume with a small convolution, and
    samples the resulting field back onto the target vertices. This is intentionally
    a practical approximation and is designed as an optional pass after the main
    surface-based transfer.

    Returns:
        target_weights: (n_target, n_bones) float32
    """
    if source_verts.shape[0] == 0:
        return np.zeros((target_verts.shape[0], source_weights.shape[1]), dtype=np.float32)
    if target_verts.shape[0] == 0:
        return np.zeros((0, source_weights.shape[1]), dtype=np.float32)

    source_weights = np.asarray(source_weights, dtype=np.float32)
    if source_weights.ndim == 1:
        source_weights = source_weights[:, np.newaxis]

    all_pts = np.vstack([source_verts, target_verts])
    bb_min = all_pts.min(axis=0)
    bb_max = all_pts.max(axis=0)
    pad = max(float((bb_max - bb_min).max() * 0.05 + 1e-4), 1e-4)
    bb_min -= pad
    bb_max += pad

    extent = bb_max - bb_min
    max_extent = max(float(extent.max()), 1e-6)
    resolution = max(2, int(resolution))
    voxel_size = max_extent / max(float(resolution - 1), 1.0)
    origin = bb_min

    target_res = np.maximum(np.ceil(extent / voxel_size).astype(int) + 1, 2)
    grid_res = np.minimum(target_res, resolution)

    n_bones = source_weights.shape[1]
    weight_grid = np.zeros((*grid_res, n_bones), dtype=np.float32)
    seed_count = np.zeros(grid_res, dtype=np.float32)

    src_idx = _world_to_grid(source_verts, origin, voxel_size, grid_res)
    for i in range(source_verts.shape[0]):
        x, y, z = src_idx[i]
        weight_grid[x, y, z] += source_weights[i]
        seed_count[x, y, z] += 1.0

    seed_mask = seed_count > 0
    if np.any(seed_mask):
        weight_grid[seed_mask] /= seed_count[seed_mask, None]

    if diffusion_steps > 0:
        kernel = np.ones((3, 3, 3), dtype=np.float32)
        kernel /= kernel.sum()
        for _ in range(int(diffusion_steps)):
            prev = weight_grid.copy()
            for b in range(n_bones):
                smoothed = ndimage.convolve(weight_grid[..., b], kernel, mode='constant', cval=0.0)
                weight_grid[..., b] = smoothed
            if np.any(seed_mask):
                weight_grid[seed_mask] = prev[seed_mask]

    if falloff != 1.0:
        seed_mask = seed_count > 0
        if np.any(seed_mask):
            dist = ndimage.distance_transform_edt(1.0 - seed_mask.astype(np.float32))
            max_dist = max(float(dist.max()), 1e-6)
            strength = 1.0 / (1.0 + max(0.0, falloff - 1.0) * dist / max_dist)
            weight_grid *= strength[..., None]

    tgt_idx = _world_to_grid(target_verts, origin, voxel_size, grid_res)
    target_weights = np.zeros((target_verts.shape[0], n_bones), dtype=np.float32)
    for i in range(target_verts.shape[0]):
        x, y, z = tgt_idx[i]
        target_weights[i] = weight_grid[x, y, z]

    if source_verts.shape[0] > 0:
        dst = target_verts[:, None, :] - source_verts[None, :, :]
        d2 = np.sum(dst * dst, axis=2)
        nearest = d2.argmin(axis=1)
        fallback = source_weights[nearest]
        target_weights = _safe_row_normalize(target_weights, fallback=fallback)
    else:
        target_weights = _safe_row_normalize(target_weights)

    if mix_factor < 1.0:
        target_weights *= mix_factor

    return target_weights.astype(np.float32)


__all__ = ["geodesic_voxel_transfer"]

"""
Notes:
- This pass is intentionally lightweight and approximate.
- It is best used as an optional post-pass after the normal surface transfer.
- For a more exact implementation, a true geodesic voxelizer or signed-distance
  field flood would be required.
"""
