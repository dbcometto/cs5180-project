# Central config for the DQN/DDQN TreeWorld experiment versions shown on the poster.
# Each version captures (1) a short description used in plot subtitles, (2) the env
# construction args used during training, and (3) any reward-constant overrides.
# Used by dqn_train / dqn_test / dqn_graph / dqn_viz so scripts stay in sync.

from treescan.environments import TreeWorld


VERSION_CONFIGS = {
    "v2": {
        "desc": "Random maps, no reward shaping — both algorithms plateau",
        "env_args": {
            "step_limit": 999,
            "use_fixed_map": False,
            "enable_extra_channels": False,
            "do_smooth_complete_reward": False,
            "discourage_early_end": False,
            "do_reward_tree_complete": False,
        },
        "rewards": {},
    },
    "v5": {
        "desc": "Fixed map with reward shaping — DDQN learns to call END",
        "env_args": {
            "step_limit": 200,
            "use_fixed_map": True,
            "enable_extra_channels": True,
            "do_smooth_complete_reward": True,
            "discourage_early_end": True,
            "do_reward_tree_complete": False,
        },
        "rewards": {"REWARD_STEP": -0.1},
    },
    "v6": {
        "desc": "Random maps with 2M training steps — advantage shrinks under generalization",
        "env_args": {
            "step_limit": 300,
            "use_fixed_map": False,
            "enable_extra_channels": True,
            "do_smooth_complete_reward": True,
            "discourage_early_end": True,
            "do_reward_tree_complete": False,
        },
        "rewards": {"REWARD_STEP": -0.1},
    },
    "v7": {
        "desc": "Boosted scan rewards on random maps — agent over-scans, END suppressed",
        "env_args": {
            "step_limit": 200,
            "use_fixed_map": False,
            "enable_extra_channels": True,
            "do_smooth_complete_reward": True,
            "discourage_early_end": True,
            "do_reward_tree_complete": True,
        },
        "rewards": {
            "REWARD_STEP": -0.1,
            "REWARD_SCAN": -0.2,
            "REWARD_NEW_FACE": 2.5,
            "REWARD_EXPLORE_TILE": 0.3,
            "REWARD_TREE_COMPLETE": 3.0,
        },
    },
}


def makeEnv(version, force_fixed_map=False):
    """Build a TreeWorld env matching the given version's training config.

    Args:
        version: key into VERSION_CONFIGS
        force_fixed_map: override use_fixed_map to True (for visualization consistency)
    """
    cfg = VERSION_CONFIGS[version]
    envArgs = dict(cfg["env_args"])
    if force_fixed_map:
        envArgs["use_fixed_map"] = True

    env = TreeWorld(render_mode=None, obs_as_tensor=True, **envArgs)
    for key, value in cfg["rewards"].items():
        setattr(env, key, value)
    return env


def versionDescription(version):
    return VERSION_CONFIGS[version]["desc"]
