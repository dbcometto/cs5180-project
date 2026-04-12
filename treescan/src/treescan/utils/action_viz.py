"""Action-distribution visualizations for TreeWorld agents.

Works for any agent exposing `choose_action(env, obs)` (DQN, DDQN, PPO, ...).
Collection is empirical — run rollouts, bin (position, action) pairs per cell.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


# matches TreeWorld.ACTIONS
ACTION_NAMES = ["WAIT", "LEFT", "RIGHT", "UP", "DOWN", "SCAN", "END"]
DIRECTIONAL = {"LEFT": 1, "RIGHT": 2, "UP": 3, "DOWN": 4}
NON_DIRECTIONAL = ["WAIT", "SCAN", "END"]


def collect_action_stats(agent, env, episodes=50, start_seed=0, greedy=True):
    """Run rollouts and count actions taken at each grid cell.

    Returns:
        counts: ndarray shape (rows, cols, num_actions)
        map_info: dict with 'hidden_map' (from last episode), 'start_location'
    """
    # try to force greedy behavior for stochastic policies
    if greedy and hasattr(agent, "policy") and hasattr(agent.policy, "epsilon"):
        originalEps = agent.policy.epsilon
        agent.policy.epsilon = 0
    else:
        originalEps = None

    rows, cols = env._map_rows, env._map_cols
    numActions = len(ACTION_NAMES)
    counts = np.zeros((rows, cols, numActions), dtype=np.int64)

    seed = start_seed
    for _ in range(episodes):
        obs, _ = env.reset(seed=seed)
        seed += 1
        done = False
        while not done:
            r, c = env._agent_location
            action = agent.choose_action(env, obs)
            counts[r, c, int(action)] += 1
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    if originalEps is not None:
        agent.policy.epsilon = originalEps

    map_info = {
        "hidden_map": env._hidden_map.copy() if env._hidden_map is not None else None,
        "start_location": tuple(env._agent_start_location),
    }
    return counts, map_info


def _cell_probs(counts):
    """Per-cell action probabilities. Unvisited cells → NaN row."""
    totals = counts.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        probs = np.where(totals > 0, counts / totals, np.nan)
    return probs, totals.squeeze(-1)


def _overlay_map(ax, map_info):
    """Draw trees (green circles), rocks (brown squares), start (blue star)."""
    if map_info is None:
        return
    hidden = map_info.get("hidden_map")
    if hidden is not None:
        rows, cols = hidden.shape
        for r in range(rows):
            for c in range(cols):
                tile = hidden[r, c]
                if tile == 1:  # TREE
                    ax.scatter(c + 0.5, r + 0.5, marker="o", s=140,
                               facecolor="#059635", edgecolor="black",
                               linewidths=1.2, zorder=5)
                elif tile == 2:  # ROCK
                    ax.scatter(c + 0.5, r + 0.5, marker="s", s=140,
                               facecolor="#584935", edgecolor="black",
                               linewidths=1.2, zorder=5)
    start = map_info.get("start_location")
    if start is not None and start[0] >= 0:
        ax.scatter(start[1] + 0.5, start[0] + 0.5, marker="*", s=220,
                   facecolor="#10C7FF", edgecolor="black",
                   linewidths=1.2, zorder=6)


def plot_directional(counts, map_info=None, ax=None,
                     title="Directional Action Probabilities",
                     cmap="viridis"):
    """Four triangles per cell, each colored by P(direction | cell)."""
    probs, visits = _cell_probs(counts)
    rows, cols, _ = counts.shape

    if ax is None:
        fig, ax = plt.subplots(figsize=(cols * 0.6 + 1.5, rows * 0.6 + 1))
    else:
        fig = ax.figure

    colormap = plt.get_cmap(cmap)
    patches = []
    colors = []

    # triangle vertices (in cell-local coords, row increases downward)
    # cell spans [c, c+1] x [r, r+1]; center at (c+0.5, r+0.5)
    # map direction -> triangle vertices
    def tri_verts(r, c, direction):
        cx, cy = c + 0.5, r + 0.5
        tl, tr_, br, bl = (c, r), (c + 1, r), (c + 1, r + 1), (c, r + 1)
        if direction == "UP":
            return [tl, tr_, (cx, cy)]
        if direction == "DOWN":
            return [bl, br, (cx, cy)]
        if direction == "LEFT":
            return [tl, bl, (cx, cy)]
        if direction == "RIGHT":
            return [tr_, br, (cx, cy)]

    for r in range(rows):
        for c in range(cols):
            if visits[r, c] == 0:
                # blank cell
                patches.append(Polygon([(c, r), (c + 1, r), (c + 1, r + 1), (c, r + 1)]))
                colors.append(np.nan)
                continue
            for name, idx in DIRECTIONAL.items():
                patches.append(Polygon(tri_verts(r, c, name)))
                colors.append(probs[r, c, idx])

    colorsArr = np.array(colors)
    pc = PatchCollection(patches, cmap=colormap, edgecolor="white", linewidth=0.5)
    pc.set_array(colorsArr)
    pc.set_clim(0, 1)
    ax.add_collection(pc)

    # gray out unvisited cells
    for r in range(rows):
        for c in range(cols):
            if visits[r, c] == 0:
                ax.add_patch(Polygon(
                    [(c, r), (c + 1, r), (c + 1, r + 1), (c, r + 1)],
                    facecolor="#DDDDDD", edgecolor="white", linewidth=0.5))

    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(cols) + 0.5)
    ax.set_xticklabels(np.arange(cols))
    ax.set_yticks(np.arange(rows) + 0.5)
    ax.set_yticklabels(np.arange(rows))
    ax.set_title(title)

    _overlay_map(ax, map_info)

    fig.colorbar(pc, ax=ax, shrink=0.8, label="P(direction | cell)")
    return fig, ax


def plot_non_directional(counts, map_info=None, title_prefix="", cmap="magma"):
    """Three heatmaps (WAIT / SCAN / END) of P(action | cell)."""
    probs, visits = _cell_probs(counts)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, name in zip(axes, NON_DIRECTIONAL):
        actionIdx = ACTION_NAMES.index(name)
        grid = probs[:, :, actionIdx]
        # mask unvisited with gray
        masked = np.ma.masked_invalid(grid)
        cmapObj = plt.get_cmap(cmap).copy()
        cmapObj.set_bad(color="#DDDDDD")
        # auto-scale to this panel's max so rare actions (like END) remain visible
        vmax = np.nanmax(grid) if np.any(grid > 0) else 1.0
        vmax = max(vmax, 1e-3)
        im = ax.imshow(masked, cmap=cmapObj, vmin=0, vmax=vmax)
        totalCalls = int(counts[:, :, actionIdx].sum())
        base = f"P({name} | cell)  [n={totalCalls}, max={vmax:.2f}]"
        ax.set_title(f"{title_prefix}{base}" if title_prefix else base)
        ax.set_xticks(np.arange(grid.shape[1]))
        ax.set_yticks(np.arange(grid.shape[0]))
        # imshow uses cell-center coords, but overlay uses cell-corner coords;
        # shift by (-0.5, -0.5) to line up
        if map_info is not None:
            hidden = map_info.get("hidden_map")
            if hidden is not None:
                for r in range(hidden.shape[0]):
                    for c in range(hidden.shape[1]):
                        tile = hidden[r, c]
                        if tile == 1:
                            ax.scatter(c, r, marker="o", s=100,
                                       facecolor="#059635", edgecolor="black",
                                       linewidths=1.0, zorder=5)
                        elif tile == 2:
                            ax.scatter(c, r, marker="s", s=100,
                                       facecolor="#584935", edgecolor="black",
                                       linewidths=1.0, zorder=5)
            start = map_info.get("start_location")
            if start is not None and start[0] >= 0:
                ax.scatter(start[1], start[0], marker="*", s=180,
                           facecolor="#10C7FF", edgecolor="black",
                           linewidths=1.0, zorder=6)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    return fig, axes


def render_action_viz(agent, env, episodes=50, start_seed=0,
                      save_dir=None, tag="", figure_suptitle=None, show=False):
    """Collect stats + render both figures. Optionally save to disk.

    Args:
        tag: appended to saved filenames, e.g. "ckpt_500000".
        figure_suptitle: bold title shown at the top of both figures
            (e.g. "DDQN — V5 @ step 500,000"). If None, uses `tag`.
    """
    counts, mapInfo = collect_action_stats(agent, env, episodes=episodes, start_seed=start_seed)

    suptitle = figure_suptitle if figure_suptitle is not None else tag

    figDir, axDir = plot_directional(counts, map_info=mapInfo,
                                     title="Directional Action Probabilities")
    if suptitle:
        figDir.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.995)
        figDir.tight_layout()

    figNonDir, _ = plot_non_directional(counts, map_info=mapInfo)
    if suptitle:
        figNonDir.suptitle(f"{suptitle} — Non-Directional Action Probabilities",
                           fontsize=13, fontweight="bold", y=1.02)
        figNonDir.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        suffix = f"_{tag}" if tag else ""
        figDir.savefig(f"{save_dir}/directional{suffix}.png", dpi=150, bbox_inches="tight")
        figNonDir.savefig(f"{save_dir}/non_directional{suffix}.png", dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(figDir)
        plt.close(figNonDir)

    return counts
