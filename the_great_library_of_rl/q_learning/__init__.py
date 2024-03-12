from abc import ABC

from torch import Tensor


class QAgent(ABC):
    """
    Interface for Q-Learning agents.
    """

    def get_q_values(self, state) -> list[float] | Tensor:
        """
        Return predicted q-values for a given state.

        Args:
            state: State of the environment.

        Returns:
            list[float] | Tensor: Q-values for a given state.
        """

        raise NotImplementedError

    def get_action(self, state) -> int:
        """
        Return the best action given a state.

        Args:
            state: State of the environment.

        Returns:
            int: Best action according to the agent.
        """

        raise NotImplementedError

    def get_q_value(self, state, action: int) -> float:
        """
        Return the q-value for a state/action pair.

        Args:
            state: State of the environment.
            action (int): Action to evaluate in the given state.

        Returns:
            float: Q-value of the state/action pair.
        """

        raise NotImplementedError

    def get_max_q_value(self, state) -> float | Tensor:
        """
        Get the maximum q-value for a given state.

        Args:
            state: State of the environment.

        Returns:
            float | Tensor: Q-value of the best action.
        """

        raise NotImplementedError

    def update_q_value(self, state, action: int, reward: float, next_state, non_terminal: bool):
        """
        Update the agent's q-value prediction.

        Args:
            state: State of the environment.
            action (int): Action taken in the state.
            reward (float): Reward experienced after taking the action.
            next_state: State reached after taking the action.
            non_terminal (bool): False if the environment ended (last state was reached), True otherwise.
        """

        raise NotImplementedError
