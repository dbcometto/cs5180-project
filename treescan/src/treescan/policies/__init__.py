"""A collection of policies"""

from .base import Policy
from .random_tabular_policy import RandomTabularPolicy
from .mc_tabular import MCTabularFirstVisitEpsilonControl
from .reinforce import DiscreteReinforce, DiscreteBatchReinforce, DiscreteBatchReinforceBaseline
from .ppo import DiscretePPO
from .a2c import DiscreteAdvantageActorCritic
from .dqn import DiscreteDQN
from .ddqn import DiscreteDoubleDQN