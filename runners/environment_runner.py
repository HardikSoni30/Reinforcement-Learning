import numpy as np
from tqdm import tqdm

from learners.algos import SARSA, ExpectedSARSA, NStepSARSA


# Class to store episode statistics like rewards and steps
class EpisodeStats:
    def __init__(self, total_episodes, n_runs):
        self.rewards = np.zeros((total_episodes, n_runs))
        self.steps = np.zeros((total_episodes, n_runs))

    def log(self, episode, run, total_rewards, step_count):
        self.rewards[episode, run] = total_rewards
        self.steps[episode, run] = step_count

class EnvironmentRunner:
    def __init__(self, env, learner, policy, params):
        self.env = env
        self.learner = learner
        self.policy = policy
        self.params = params

        # Access total_episodes and n_runs
        self.total_episodes = self.params.sim_params.total_episodes
        self.n_runs = self.params.sim_params.n_runs
        self.state_size = self.params.learner_params.state_size
        self.action_size = self.params.learner_params.action_size
        self.seed = self.params.sim_params.seed

    def run_env(self):
        # Initialize statistics and tracking variables
        stats = EpisodeStats(self.total_episodes, self.n_runs)
        episodes = np.arange(self.total_episodes)
        q_tables = np.zeros((self.n_runs, self.state_size, self.action_size))
        all_states = []
        all_actions = []

        # Run the environment multiple times (for n_runs)
        for run in range(self.n_runs):
            self.learner.reset_qtable()  # Reset the Q-table between runs

            # Iterate over each episode
            for episode in tqdm(episodes, desc=f"Run {run+1}/{self.n_runs} - Episodes", leave=False):
                total_rewards, step_count, states, actions = self.run_single_episode()

                # Log the states and actions
                all_states.extend(states)
                all_actions.extend(actions)

                # Log statistics for the episode
                stats.log(episode, run, total_rewards, step_count)

            # Store the final Q-table for the run
            q_tables[run, :, :] = self.learner.qtable

        return stats.rewards, stats.steps, episodes, q_tables, all_states, all_actions

    def run_single_episode(self):
        state = self.env.reset(seed=self.seed)[0]
        done = False
        total_rewards = 0
        step_count = 0
        states, actions, rewards = [], [], []

        # Choose initial action (useful for SARSA-like methods)
        action = self.policy.choose_action(
            action_space=self.env.action_space,
            state=state, qtable=self.learner.qtable,
            t=step_count
        )

        while not done:
            # Log current state and action
            states.append(state)
            actions.append(action)

            # Take the action and observe the outcome
            new_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            rewards.append(reward)

            # For SARSA, Expected SARSA, and N-step SARSA: choose next action at new_state
            if isinstance(self.learner, SARSA) or isinstance(self.learner, ExpectedSARSA):
                new_action = self.policy.choose_action(
                    action_space=self.env.action_space,
                    state=new_state, qtable=self.learner.qtable,
                    t=step_count
                )
                # SARSA: update the Q-table using the new action
                self.learner.update(state, action, reward, new_state, new_action)
                action = new_action  # Set next action

            elif isinstance(self.learner, NStepSARSA):
                # For N-step SARSA, update when enough steps are gathered
                if len(states) >= self.learner.n_steps:
                    self.learner.update(states[-self.learner.n_steps:], actions[-self.learner.n_steps:],
                                        rewards[-self.learner.n_steps:])

            elif isinstance(self.learner, ExpectedSARSA):
                # Expected SARSA: update based on expectation of future actions
                self.learner.update(state, action, reward, new_state)

            else:
                # Q-learning, Double Q-learning, etc.
                self.learner.update(state, action, reward, new_state)

            # Move to the next state
            state = new_state

            # Choose next action for Q-learning-like methods
            if not (isinstance(self.learner, SARSA) or isinstance(self.learner, NStepSARSA)):
                action = self.policy.choose_action(
                    action_space=self.env.action_space,
                    state=state, qtable=self.learner.qtable,
                    t=step_count
                )

            # Update totals
            total_rewards += reward
            step_count += 1

        return total_rewards, step_count, states, actions
