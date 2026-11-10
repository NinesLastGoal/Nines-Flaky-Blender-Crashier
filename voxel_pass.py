# voxel_pass.py
# Geodesic-style volumetric weight transfer pass for Robust Weight Transfer
# Simplified volume diffusion that lets weights flow through space / across gaps.
# Suitable for armor plates, chain-link style pieces, and disconnected geometry.

import numpy as np
from scipy import ndimage

def _world_to_grid(points, origin, voxel_size, resolution):
    """Convert world points to grid indices (clipped)."""
    idx = np.floor((points - origin) / voxel_size).astype(np.int32)
    idx = np.clip(idx, 0, resolution - 1)
    return idx

def geodesic_voxel_transfer(
    source_verts: np.ndarray,      # (Ns, 3)
    source_weights: np.ndarray,    # (Ns, Nb)
    target_verts: np.ndarray,      # (Nt, 3)
    resolution: int = 64,
    diffusion_steps: int = 8,
    falloff: float = 1.0,
    mix_factor: float = 1.0,
):
    """
    Volume-based weight transfer.
    Weights are seeded from source vertices into a regular grid,
    diffused through the volume, then sampled back onto target vertices.

    Returns:
        target_weights: (Nt, Nb) float32
    """
    if source_verts.shape[0] == 0 or target_verts.shape[0] == 0:
        return np.zeros((target_verts.shape[0], source_weights.shape[1]), dtype=np.float32)

    # Combined AABB with padding
    all_pts = np.vstack([source_verts, target_verts])
    bb_min = all_pts.min(axis=0)
    bb_max = all_pts.max(axis=0)
    pad = (bb_max - bb_min).max() * 0.05 + 1e-4
    bb_min -= pad
    bb_max += pad

    extent = bb_max - bb_min
    max_extent = extent.max()
    voxel_size = max_extent / max(resolution - 1, 1)
    origin = bb_min

    # Grid shape (roughly cubic)
    grid_res = np.maximum(np.ceil(extent / voxel_size).astype(int) + 1, 2)
    grid_res = np.minimum(grid_res, resolution)

    Nb = source_weights.shape[1]
    weight_grid = np.zeros((*grid_res, Nb), dtype=np.float32)
    count_grid = np.zeros(grid_res, dtype=np.float32)

    # Seed source weights into voxels
    src_idx = _world_to_grid(source_verts, origin, voxel_size, grid_res)
    for i in range(src_idx.shape[0]):
        x, y, z = src_idx[i]
        weight_grid[x, y, z] += source_weights[i]
        count_grid[x, y, z] += 1.0

    # Average where multiple sources hit the same voxel
    mask = count_grid > 0
    weight_grid[mask] /= count_grid[mask, None]

    # Diffuse through volume (simple iterative averaging = cheap geodesic-like flow)
    # This lets weights spread across gaps and into disconnected islands.
    kernel = np.ones((3, 3, 3), dtype=np.float32)
    kernel /= kernel.sum()

    for _ in range(max(1, diffusion_steps)):
        for b in range(Nb):
            weight_grid[..., b] = ndimage.convolve(weight_grid[..., b], kernel, mode='nearest')

    # Optional falloff (softens long-range influence)
    if falloff < 1.0:
        # Approximate distance from original seeds
        seed_mask = (count_grid > 0).astype(np.float32)
        dist = ndimage.distance_transform_edt(1.0 - seed_mask)
        dist_norm = dist / (dist.max() + 1e-6)
        falloff_mask = np.power(1.0 - dist_norm, falloff)
        weight_grid *= falloff_mask[..., None]

    # Sample at target vertices (nearest for speed; trilinear optional later)
    tgt_idx = _world_to_grid(target_verts, origin, voxel_size, grid_res)
    target_weights = np.zeros((target_verts.shape[0], Nb), dtype=np.float32)
    for i in range(tgt_idx.shape[0]):
        x, y, z = tgt_idx[i]
        target_weights[i] = weight_grid[x, y, z]

    # Normalize rows
    row_sums = target_weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    target_weights /= row_sums

    # Mix factor (1.0 = full replace)
    if mix_factor < 1.0:
        # Caller can blend; here we just scale for simplicity
        target_weights *= mix_factor

    return target_weights.astype(np.float32)
