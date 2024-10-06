# Reinforcement Learning Algorithms

This repository contains implementations of various reinforcement learning algorithms, including Q-learning, SARSA, Expected SARSA, Double Q-learning, and n-step SARSA. These algorithms have been tested on the FrozenLake environment from the OpenAI Gymnasium library.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Algorithms](#algorithms)
  - [Q-learning](#q-learning)
  - [SARSA](#sarsa)
  - [Expected SARSA](#expected-sarsa)
  - [Double Q-learning](#double-q-learning)
  - [n-step SARSA](#n-step-sarsa)
- [Usage](#usage)
- [Results](#results)
- [Motivation](#motivation)
- [License](#license)

## Introduction

Reinforcement learning (RL) is a machine learning paradigm where an agent learns to make decisions by interacting with an environment. The goal is to maximize cumulative rewards through trial and error. This project focuses on several popular RL algorithms implemented and tested on the FrozenLake environment, a grid-based environment where the agent must navigate to a goal while avoiding holes.

## Installation

You can install the Gymnasium library with the FrozenLake environment as follows:

```bash
pip install gymnasium[classic-control]
```

## Algorithms

### Q-learning

Q-learning is an off-policy algorithm that learns the value of an action in a given state. It uses the Bellman equation to update the action-value function based on the agent's experiences.

### SARSA

SARSA (State-Action-Reward-State-Action) is an on-policy algorithm that updates the action-value function based on the action actually taken by the agent, making it more sensitive to the policy being followed.

### Expected SARSA

Expected SARSA is a variation of SARSA that uses the expected value of the next state-action pair rather than the actual next action. This approach can lead to more stable updates.

### Double Q-learning

Double Q-learning addresses the overestimation bias of Q-learning by maintaining two separate value functions. Updates are performed using one value function to select actions and the other to evaluate them.

### n-step SARSA

n-step SARSA is a generalization of the SARSA algorithm that considers multiple time steps in updating the action-value function. This allows the agent to leverage more information from its experiences.

## Usage

To demonstrate the usage of these algorithms on the FrozenLake environment, refer to the `frozen_lake_train.ipynb` Jupyter Notebook. This notebook provides a step-by-step guide for training and evaluating the algorithms, including visualizations of the results.

### Running the Notebook

1. Open `frozen_lake_train.py` and follow the instructions to run the cells.

## Results

The results for each algorithm, including performance metrics and visualizations, are presented within the notebook. Key metrics include:

- Average reward per episode
- Convergence time (number of episodes)
- Performance comparison between different algorithms

## Motivation

The motivation for this project is drawn from the [Gymnasium FrozenLake tutorial]([https://www.gymlibrary.com](https://gymnasium.farama.org/tutorials/training_agents/FrozenLake_tuto/#sphx-glr-tutorials-training-agents-frozenlake-tuto-py)/), which provides a foundational understanding of practical implementations. This repository aims to extend that knowledge by implementing several algorithms and evaluating their performance in a controlled environment.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI Gym for the FrozenLake environment.
- Reinforcement learning literature and resources.

Feel free to explore the code, run experiments, and contribute to improving the algorithms!
```
