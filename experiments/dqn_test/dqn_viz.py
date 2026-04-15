# generates action-distribution visualizations at selected checkpoints
# for both dqn and ddqn for a given version

import os
import re
import shutil

from treescan.agents import Agent
from treescan.utils import render_action_viz
from version_configs import makeEnv, versionDescription


agentsFolderpath = "/Users/adamlewis/Desktop/Northeastern/Reinforcement Learning/Project/cs5180-project/experiments/dqn_test/agents"

version = "v8"
vizEpisodes = 50
vizStartSeed = 5000
numSnapshots = 5  # evenly-spaced checkpoints to visualize


def checkpointSteps(folderpath):
    """find all checkpoint step numbers saved in folderpath/checkpoints/."""
    ckptDir = f"{folderpath}/checkpoints"
    if not os.path.isdir(ckptDir):
        return []
    steps = []
    for name in os.listdir(ckptDir):
        m = re.match(r"ckpt_(\d+)\.pt$", name)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


print(f"\nVisualizing {version.upper()} — {versionDescription(version)}\n")

for algo in ["dqn", "ddqn"]:
    folderpath = f"{agentsFolderpath}/{algo}_{version}"
    steps = checkpointSteps(folderpath)
    if not steps:
        print(f"no checkpoints for {algo}_{version}, skipping")
        continue

    # evenly subsample checkpoints (always keep first and last)
    if len(steps) > numSnapshots:
        idxs = [round(i * (len(steps) - 1) / (numSnapshots - 1)) for i in range(numSnapshots)]
        steps = [steps[i] for i in sorted(set(idxs))]

    print(f"{algo.upper()} {version}: visualizing {len(steps)} checkpoints → {steps}")
    vizDir = f"{folderpath}/viz"

    # clear previous viz run so the folder only contains this run's images
    if os.path.isdir(vizDir):
        shutil.rmtree(vizDir)

    for step in steps:
        agent = Agent.load_from_checkpoint(folderpath, step)
        # use fixed map for viz so map overlay is consistent across checkpoints
        env = makeEnv(version, force_fixed_map=True)
        tag = f"{algo.upper()}_{version}_ckpt_{step}"
        print(f"  {tag}")
        render_action_viz(agent, env, episodes=vizEpisodes,
                          start_seed=vizStartSeed, save_dir=vizDir, tag=tag,
                          figure_suptitle=f"{algo.upper()} — {version.upper()} @ step {step:,}")

    print(f"  saved to {vizDir}\n")
