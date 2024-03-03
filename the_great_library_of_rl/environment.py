from abc import ABC, abstractmethod


class Environment(ABC):
    @abstractmethod
    def reset(self) -> None:
        """
        Reset the state of the environment to its initial value.
        """

        pass

    @abstractmethod
    def get_state(self):
        """
        Return the current state of the environment.

        Returns:
            any: Information about the state of the environment.
        """

        pass

    @abstractmethod
    def get_action_count(self) -> int:
        """
        Return the number of possible actions.

        Returns:
            int: The number of possible actions at any given moment.
        """

        pass

    @abstractmethod
    def step(self, action: int) -> None:
        """
        Execute an action in the environment.

        Args:
            action (int): Action to take in the environment.
        """

        pass

    @abstractmethod
    def get_reward(self) -> float:
        """
        Return the last observed reward.

        Returns:
            float: Reward after the last step.
        """

        pass

    @abstractmethod
    def is_terminated(self) -> bool:
        """
        Check if the current state is terminal. That means checking if the environment ended.

        Returns:
            bool: True if the current state is terminal, False otherwise.
        """

        pass

    @abstractmethod
    def close(self) -> None:
        """
        Close and cleanup after the environment.
        """

        pass

    @abstractmethod
    def set_evaluation(self, value: bool) -> None:
        """
        Change if the environment is in an evaluation or training phase.

        Args:
            value: True to set the environment to evaluation, False otherwise.
        """

        pass
