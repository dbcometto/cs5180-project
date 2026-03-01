"""A collection of policies"""

from .base import Policy
from .random_tabular_policy import RandomTabularPolicy
from .mc_tabular import MCTabularFirstVisitEpsilonControl
from .reinforce import DiscreteReinforce
from .ppo import DiscretePPO