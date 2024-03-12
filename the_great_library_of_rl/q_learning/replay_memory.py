from random import sample

from torch import Tensor, stack, tensor


class Experience:
    """
    An agent's experience from the environment.

    Args:
        state (Tensor): State the agent was in.
        action (int): Action that the agent took.
        reward (float): Received/Observed reward.
        next_state (Tensor): New state after taking the action.
        non_terminal (bool): False if the environment ended (last state was reached), True otherwise.
    """

    def __init__(self, state: Tensor, action: int, reward: float, next_state: Tensor, non_terminal: bool) -> None:
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.non_terminal = non_terminal


class ExperienceBatch:
    """
    A batch of experience from a replay memory. Extracts states, rewards, actions, and next states from the
    experience objects into tensors ready for training.
    """

    def __init__(self, experiences: list[Experience]) -> None:
        states = []
        actions = []
        rewards = []
        next_states = []
        non_terminal = []

        # OPTIMIZATION NOTE: A loop like this may not be optimal.
        for e in experiences:
            states.append(e.state)
            actions.append(e.action)
            rewards.append(e.reward)
            next_states.append(e.next_state)
            non_terminal.append(1 if e.non_terminal else 0)

        self.states = stack(states)
        self.actions = tensor(actions)
        self.rewards = tensor(rewards)
        self.next_states = stack(next_states)
        self.non_terminal = tensor(non_terminal)


class ReplayMemory:
    """
    Agent's memory containing experiences. Used to provide more stable and effective training.

    Args:
        capacity (int): Maximum size of the memory. Exceeding it will delete the oldest records.
        update_after_episodes (int): Number of episodes until a parameter update.
        batch_size (int): Number of experiences to sample.
    """

    def __init__(self, capacity: int, update_after_episodes: int, batch_size: int) -> None:
        self.capacity = capacity
        self.update_after_episodes = update_after_episodes
        self.batch_size = batch_size
        self.experiences = []

    def add_experience(self, e: Experience) -> None:
        """
        Add a new experience to the replay memory. The oldest record will be deleted if the capacity is exceeded.

        Args:
            e: Experience to add.
        """

        self.experiences.append(e)

        if self.capacity < len(self.experiences):
            self.experiences.pop(0)

    def sample_batch(self, batch_size_overwrite: int = None) -> ExperienceBatch:
        """
        Return a list of random experiences from the memory (random without replacement).
        The whole list of experiences is returned if the batch size exceeds its length.

        Args:
            batch_size_overwrite (int): Number of experiences to sample. (Overwrites the object's batch size property.)

        Returns:
            ExperienceBatch: An object containing the batch of experiences.
        """

        batch_size = self.batch_size if batch_size_overwrite is None else batch_size_overwrite

        if batch_size >= len(self.experiences):
            return ExperienceBatch(self.experiences)

        return ExperienceBatch(sample(self.experiences, batch_size))
