from pathlib import Path

import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from utils import postprocess
from config import Params, EpsilonParams, EnvironmentParams, LearnerParams, SimulationParams
from policies.epsilon_policies import EpsilonGreedy
from learners.algos import Qlearning, SARSA
from runners.environment_runner import EnvironmentRunner

#%%
env_params = EnvironmentParams(params={
    "map_size": 4,
    "is_slippery": False,
    "proba_frozen": 0.9,
    "seed": 123
})
learner_params = LearnerParams(learning_rate=0.8, gamma=0.95, action_size=0, state_size=0)
sim_params = SimulationParams(total_episodes=2000, n_runs=20, seed=123, savefig_folder=Path("../imgs/"))
params = Params(env_params=env_params, learner_params=learner_params, sim_params=sim_params)
#%%
env_fl = gym.make(
    "FrozenLake-v1",
    is_slippery=env_params.params.get("is_slippery", False),
    render_mode="rgb_array",
    desc=generate_random_map(
        size=env_params.params.get("map_size"),
        p=env_params.params.get("proba_frozen"),
        seed=env_params.params.get("seed")
    ),
)
#%%
learner_params.action_size = env_fl.action_space.n
learner_params.state_size = env_fl.observation_space.n
# params = params._replace(action_size=env_fl.action_space.n)
# params = params._replace(state_size=env_fl.observation_space.n)

print(f"Action size: {learner_params.action_size}")
print(f"State size: {learner_params.state_size}")

#%%
# learner = Qlearning(
#     learning_rate=learner_params.learning_rate,
#     gamma=learner_params.gamma,
#     state_size=learner_params.state_size,
#     action_size=learner_params.action_size,
# )
learner = SARSA(
    learning_rate=learner_params.learning_rate,
    gamma=learner_params.gamma,
    state_size=learner_params.state_size,
    action_size=learner_params.action_size
)
#%%
# epsilon_params_constant = EpsilonParams(
#     begin_value=0.1, end_value=0.1, linear=False
# )
# # Using constant schedule
# epsilon_greedy_constant = EpsilonGreedy(
#     epsilon_params_constant,
#     begin_t=0,
#     end_t=201)
epsilon_params_linear = EpsilonParams(begin_value=1.0, end_value=0.1, linear=True)
# Using linear schedule
epsilon_greedy_linear = EpsilonGreedy(epsilon_params_linear, begin_t=0, end_t=201)
#%%
runner = EnvironmentRunner(env_fl, learner, epsilon_greedy_linear, params)

#%%
rewards, steps, episodes, q_tables, all_states, all_actions = runner.run_env()
env_fl.close()
#%%
detailed_df, summary_df = postprocess(episodes,
                                      params.sim_params.n_runs,
                                      rewards,
                                      steps,
                                      params.env_params.params.get("map_size"))

