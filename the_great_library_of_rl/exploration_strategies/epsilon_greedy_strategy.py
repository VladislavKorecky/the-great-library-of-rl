from random import random


class EpsilonGreedyStrategy:
    """
    Epsilon-greedy is an exploration strategy that tries to balance the exploration and exploitation with a variable
    epsilon. Epsilon is a number from 0 to 1 that indicates the percentage of random actions compared to the agent's
    action. 1 means 100% of random actions. This value usually starts at 1 and slowly decreases over-time.

    Args:
        epsilon_start (float): Starting value for epsilon.
        epsilon_end (float): Minimum value for epsilon.
        epsilon_step (float): How much epsilon decreases when it decays.
    """

    def __init__(self, epsilon_start: float, epsilon_end: float, epsilon_step: float) -> None:
        self.epsilon_end = epsilon_end
        self.epsilon_step = epsilon_step

        self.epsilon = epsilon_start

    def should_sample_random_action(self) -> bool:
        """
        Determine if the agent should explore the environment by sampling a random action or exploit the environment
        by picking the best action.

        Returns:
            bool: True if the agent should explore, False if it should exploit.
        """

        return random() < self.epsilon

    def decay_epsilon(self) -> None:
        """
        Decrease epsilon by the defined epsilon step and clip it to the minimum value (epsilon end).
        """

        self.epsilon -= self.epsilon_step

        if self.epsilon < self.epsilon_end:
            self.epsilon = self.epsilon_end
