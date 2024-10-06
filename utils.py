import numpy as np
import pandas as pd
import plotly.graph_objects as go

def linear_schedule(begin_value, end_value, begin_t, end_t=None, decay_steps=None):
    """Linear schedule, used for exploration epsilon in DQN agents."""
    decay_steps = decay_steps if end_t is None else end_t - begin_t

    def step(t):
        """Implements a linear transition from a begin to an end value."""
        frac = min(max(t - begin_t, 0), decay_steps) / decay_steps
        return (1 - frac) * begin_value + frac * end_value

    return step


def constant_schedule(end_value):
    """Constant epsilon schedule."""
    def step(t):
        """Returns the beginning value for all time steps."""
        return end_value
    return step


def plot_cumulative_rewards(rewards_per_episode, title='frozen_lake', window_size=100):
    """
    Plot the cumulative sum of rewards over episodes using Plotly.

    Parameters:
    - rewards_per_episode (numpy.ndarray): Array of rewards per episode.
    - window_size (int): The size of the window for the rolling sum.
    """
    smoothed_rewards = np.convolve(rewards_per_episode, np.ones(window_size), mode='valid') / window_size
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=np.arange(len(smoothed_rewards)),
        y=smoothed_rewards,
        mode='lines',
        name='Smoothed Reward'
    ))

    fig.update_layout(
        title='Smoothed Rewards Over Episodes',
        xaxis_title='Episode',
        yaxis_title='Smoothed Reward',
        template='plotly_dark'
    )

    fig.write_html(title+'.html')
    fig.show()


def postprocess(episodes, n_runs, rewards, steps, map_size):
    """Convert the results of the simulation into DataFrames.

    Args:
        episodes (array-like): List of episode indices.
        n_runs (int): Number of times the environment is run.
        rewards (ndarray): 2D array of rewards for each episode and run.
        steps (ndarray): 2D array of steps for each episode and run.
        map_size (int): Size of the map, used for labeling the output.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames, one for detailed results
        and one for summary statistics.
    """
    # Create detailed results DataFrame
    detailed_df = pd.DataFrame({
        "Episodes": np.tile(episodes, reps=n_runs),  # Repeat episodes for each run
        "Rewards": rewards.flatten(order="F"),  # Flatten rewards by columns (runs)
        "Steps": steps.flatten(order="F"),  # Flatten steps by columns (runs)
    })

    # Cumulative rewards over episodes and runs
    detailed_df["cum_rewards"] = rewards.cumsum(axis=0).flatten(order="F")

    # Add map size as a repeated column
    detailed_df["map_size"] = np.repeat(f"{map_size}x{map_size}", detailed_df.shape[0])

    # Calculate success rate for each episode (Assume success when reward > 0)
    success_rate = (rewards > 0).mean(axis=1)  # Success rate over all runs for each episode

    # Create summary statistics DataFrame (mean steps per episode and success rate)
    summary_df = pd.DataFrame({
        "Episodes": episodes,  # Episode numbers
        "Steps": steps.mean(axis=1),  # Mean steps over runs for each episode
        "Success Rate": success_rate  # Success rate per episode
    })

    return detailed_df, summary_df
