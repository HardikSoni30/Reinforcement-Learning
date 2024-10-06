from typing import NamedTuple

import numpy as np

from dataclasses import dataclass
from pathlib import Path

@dataclass
class EnvironmentParams:
    params: dict

@dataclass
class LearnerParams:
    learning_rate: float       # Learning rate
    gamma: float               # Discounting rate
    action_size: int           # Number of possible actions
    state_size: int            # Number of possible states

@dataclass
class SimulationParams:
    total_episodes: int        # Total episodes
    n_runs: int                # Number of runs
    seed: int                  # Define a seed for reproducible results
    savefig_folder: Path       # Root folder where plots are saved

@dataclass
class Params:
    env_params: EnvironmentParams   # Environment-related parameters
    learner_params: LearnerParams    # Learner-related parameters
    sim_params: SimulationParams      # Simulation-related parameters

# Example instantiation
# env_params = EnvironmentParams(params={
#     "map_size": 10,
#     "is_slippery": True,
#     "proba_frozen": 0.9,
#     "additional_param": "value"
# })
# learner_params = LearnerParams(learning_rate=0.01, gamma=0.99, action_size=4, state_size=100)
# sim_params = SimulationParams(total_episodes=1000, n_runs=10, seed=42, savefig_folder=Path('results'))

# params = Params(env_params=env_params, learner_params=learner_params, sim_params=sim_params)

class EpsilonParams(NamedTuple):
    begin_value: float
    end_value: float
    linear: bool



