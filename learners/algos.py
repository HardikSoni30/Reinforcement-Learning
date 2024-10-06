from abc import ABC, abstractmethod

import numpy as np


class BaseLearner(ABC):
    def __init__(self, learning_rate, gamma, state_size, action_size):
        self.qtable = None
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.reset_qtable()

    @abstractmethod
    def update(self, *args):
        """Update the Q-table with the specific update rule."""
        pass

    @abstractmethod
    def reset_qtable(self):
        """Reset the Q-table."""
        pass


class Qlearning(BaseLearner):
    def update(self, state, action, reward, new_state):
        """Q-learning update rule."""
        delta = (
            reward
            + self.gamma * np.max(self.qtable[new_state, :])
            - self.qtable[state, action]
        )
        self.qtable[state, action] += self.learning_rate * delta

    def reset_qtable(self):
        self.qtable = np.zeros((self.state_size, self.action_size))


class SARSA(BaseLearner):
    def update(self, state, action, reward, new_state, new_action):
        """SARSA update rule."""
        td_target = reward + self.gamma * self.qtable[new_state, new_action]
        td_error = td_target - self.qtable[state, action]
        self.qtable[state, action] += self.learning_rate * td_error

    def reset_qtable(self):
        self.qtable = np.zeros((self.state_size, self.action_size))


class DoubleQlearning(BaseLearner):
    def __init__(self, learning_rate, gamma, state_size, action_size):
        super().__init__(learning_rate, gamma, state_size, action_size)
        self.qtable2 = None
        self.qtable1 = None
        self.reset_qtable()

    def update(self, state, action, reward, new_state):
        if np.random.uniform(0, 1) < 0.5:
            # Update Q1
            a_prime = np.argmax(self.qtable1[state, :])
            self.qtable1[state, action] += self.learning_rate * (
                reward + self.gamma * self.qtable2[new_state, a_prime] - self.qtable1[state, action]
            )
        else:
            # Update Q2
            a_prime = np.argmax(self.qtable2[state, :])
            self.qtable2[state, action] += self.learning_rate * (
                reward + self.gamma * self.qtable1[new_state, a_prime] - self.qtable2[state, action]
            )
        self.qtable = (self.qtable1 + self.qtable2) / 2

    def reset_qtable(self):
        self.qtable = np.zeros((self.state_size, self.action_size))
        self.qtable1 = np.zeros((self.state_size, self.action_size))
        self.qtable2 = np.zeros((self.state_size, self.action_size))


class ExpectedSARSA(BaseLearner):
    def __init__(self, learning_rate, gamma, state_size, action_size, epsilon=0.2):
        super().__init__(learning_rate, gamma, state_size, action_size)
        self.epsilon = epsilon

    def update(self, state, action, reward, new_state):
        """Expected SARSA update rule."""
        # Calculate expected value over all actions in the next state
        n_actions = self.qtable.shape[1]
        action_probs = np.ones(n_actions) * self.epsilon / n_actions
        best_next_action = np.argmax(self.qtable[new_state, :])
        action_probs[best_next_action] += (1 - self.epsilon)

        expected_q = np.sum(self.qtable[new_state, :] * action_probs)

        # Update the Q-table
        td_target = reward + self.gamma * expected_q
        td_error = td_target - self.qtable[state, action]
        self.qtable[state, action] += self.learning_rate * td_error

    def reset_qtable(self):
        self.qtable = np.zeros((self.state_size, self.action_size))


class NStepSARSA(BaseLearner):
    def __init__(self, learning_rate, gamma, state_size, action_size, n_steps=3):
        super().__init__(learning_rate, gamma, state_size, action_size)
        self.n_steps = n_steps

    def update(self, states, actions, rewards):
        """N-step SARSA update rule."""
        G = 0
        for t in range(len(rewards)):
            G += rewards[t] * (self.gamma ** t)

        state, action = states[0], actions[0]
        self.qtable[state, action] += self.learning_rate * (G - self.qtable[state, action])

    def reset_qtable(self):
        self.qtable = np.zeros((self.state_size, self.action_size))

