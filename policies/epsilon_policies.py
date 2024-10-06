import numpy as np

from config import EpsilonParams
from utils import linear_schedule, constant_schedule


class EpsilonGreedy:
    def __init__(self, epsilon_params: EpsilonParams, begin_t: int, end_t: int):
        self.epsilon = None
        self.epsilon_params = epsilon_params
        if epsilon_params.linear:
            self.epsilon_schedule = linear_schedule(
                epsilon_params.begin_value,
                epsilon_params.end_value,
                begin_t,
                end_t
            )
        else:
            self.epsilon_schedule = constant_schedule(epsilon_params.begin_value)

    def choose_action(self, action_space, state, qtable, t):
        """Choose an action `a` in the current world state (s)."""
        self.epsilon = self.epsilon_schedule(t)

        # Random number for exploration vs exploitation
        explor_exploit_tradeoff = np.random.uniform(0, 1)

        # Exploration
        if explor_exploit_tradeoff < self.epsilon:
            action = action_space.sample()
        else:
            # Exploitation (taking the action with the highest Q-value)
            values_ = qtable[state, :]
            action = np.random.choice([action_ for action_, value_ in
                                       enumerate(values_)
                                       if value_ == np.max(values_)])
        return action



class EpsilonSoftPolicy:
    def __init__(self, epsilon=0.2):
        self.epsilon = epsilon

    def choose_action(self, qtable, state):
        """Choose an action using an epsilon-soft policy."""
        n_actions = len(qtable[state])
        # Initialize probabilities with uniform distribution
        probabilities = np.ones(n_actions) * self.epsilon / n_actions

        # Get the Q-values for the current state
        values_ = qtable[state]
        # Find the best action(s)
        best_action = np.random.choice([action_ for action_, value_ in
                                        enumerate(values_)
                                        if value_ == np.max(values_)])
        # Update the probability for the best action
        probabilities[best_action] += (1.0 - self.epsilon)

        # Choose an action based on the computed probabilities
        return np.random.choice(np.arange(n_actions), p=probabilities)
