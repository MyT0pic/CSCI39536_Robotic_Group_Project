"""Frontiers: free cells next to unexplored space, grouped into goals for exploring."""
from dataclasses import dataclass
from typing import List
import numpy as np
from scipy import ndimage
from robot.mapping.grid_map import GridMap


@dataclass
class Frontier:
    cells: np.ndarray  # (N, 2) grid cells (i, j) along this stretch of frontier

    @property
    def size(self) -> int:
        return len(self.cells)


def unexplored(state: np.ndarray, min_unknown: int = 50) -> np.ndarray:
    """Unknown cells in patches of at least min_unknown cells (smaller ones are gaps between beams)."""
    patches, _ = ndimage.label(state == GridMap.UNKNOWN)
    patch_size = np.bincount(patches.reshape(-1))
    return (patches > 0) & (patch_size[patches] >= min_unknown)


def find_frontiers(state: np.ndarray, min_size: int = 8, min_unknown: int = 50, piece: int = 20,
                   wall_margin: int = 3, fill_gaps: int = 2) -> List[Frontier]:
    """Groups of free cells touching unexplored space, at least min_size cells each.

    Args:
        state: grid of UNKNOWN / FREE / BLOCKED.
        min_size: smallest frontier (cells).
        min_unknown: smallest unknown patch that counts as unexplored (cells).
        piece: long frontiers are cut into pieces of about this many cells across, so each part can be a goal.
        wall_margin: unknown cells this close (cells) to an obstacle don't count (space behind walls, gaps
            seen at a grazing angle).
        fill_gaps: unknown gaps inside the explored area up to about twice this (cells) across count as
            explored (morphological closing).
    """
    if fill_gaps > 0:
        known = state != GridMap.UNKNOWN
        gaps = ndimage.binary_closing(known, structure=np.ones((3, 3)), iterations=fill_gaps) & ~known
        state = np.where(gaps, GridMap.FREE, state).astype(state.dtype)
    free = state == GridMap.FREE
    # Unknown cells right next to an obstacle are usually just the space behind it: driving up to them
    # reveals nothing, so they don't count
    near_obstacle = state == GridMap.BLOCKED
    if wall_margin > 0:
        near_obstacle = ndimage.binary_dilation(near_obstacle, structure=np.ones((3, 3)), iterations=wall_margin)
    unknown = unexplored(state, min_unknown) & ~near_obstacle
    edge = free & ndimage.binary_dilation(unknown)  # 4-connected: shares a side with unknown
    labels, count = ndimage.label(edge, structure=np.ones((3, 3)))
    if count == 0:
        return []
    sizes = np.bincount(labels.reshape(-1))
    labels[sizes[labels] < min_size] = 0  # whole frontiers too small to bother with
    cells = np.argwhere(labels > 0)
    if len(cells) == 0:
        return []
    tiles = cells // piece
    groups = (labels[cells[:, 0], cells[:, 1]].astype(np.int64) * 1_000_000 + tiles[:, 0] * 1000 + tiles[:, 1])
    order = np.argsort(groups, kind="stable")
    cells, groups = cells[order], groups[order]
    splits = np.flatnonzero(np.diff(groups)) + 1
    return [Frontier(c) for c in np.split(cells, splits)]
