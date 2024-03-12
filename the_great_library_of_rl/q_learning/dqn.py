from torch import Tensor, argmax, tensor, arange
from torch import max as torch_max
from torch.nn.functional import mse_loss
from torch import float as torch_float

from the_great_library_of_rl.neural_network import NeuralNetwork
from the_great_library_of_rl.q_learning import QAgent
from the_great_library_of_rl.q_learning.replay_memory import Experience, ReplayMemory


class DQN(QAgent):
    """
    Q-Learning agent using neural networks to approximate Q-values for a given state.

    Args:
        network (NeuralNetwork): Neural network to use.
        gamma (float): Decay rate for future rewards.
    """

    def __init__(self, network: NeuralNetwork, gamma: float, replay_memory: ReplayMemory = None):
        self.network = network
        self.gamma = gamma
        self.replay_memory = replay_memory

        # number of elapsed episodes (counted in the update method)
        self._episode = 0

    def get_q_values(self, state: Tensor) -> Tensor:
        """
        Return predicted q-values for a given state.

        Args:
            state (Tensor): State of the environment.

        Returns:
            Tensor: Q-values for a given state.
        """

        return self.network.forward(state)

    def get_action(self, state) -> int:
        """
        Return the best action given a state.

        Args:
            state: State of the environment.

        Returns:
            int: Best action according to the agent.
        """

        return argmax(self.get_q_values(state)).item()

    def get_q_value(self, state, action: int) -> float:
        """
        Return the q-value for a state/action pair.

        Args:
            state: State of the environment.
            action (int): Action to evaluate in the given state.

        Returns:
            float: Q-value of the state/action pair.
        """

        return self.get_q_values(state)[action].item()

    def get_max_q_value(self, state) -> Tensor:
        """
        Get the maximum q-value for a given state.

        Args:
            state: State of the environment.

        Returns:
            Tensor: Q-value of the best action.
        """

        return torch_max(self.get_q_values(state), dim=-1).values

    def update_q_value(self, state, action: int, reward: float, next_state, non_terminal: bool) -> None:
        """
        Add a new experience to the replay memory. Update the DQN's parameters to better approximate the Q values.

        Args:
            state: State of the environment.
            action: Action taken by the agent.
            reward: Received/Observed reward.
            next_state: New state after taking the action.
            non_terminal (bool): False if the environment ended (last state was reached), True otherwise.
        """

        self._episode += 1

        # update without a replay memory
        if self.replay_memory is None:
            q = self.get_q_values(state)[action]
            target_q = reward + self.gamma * self.get_max_q_value(next_state) if non_terminal else 0

            optimizer = self.network.get_optimizer()

            optimizer.zero_grad()
            loss = mse_loss(q, target_q)
            loss.backward()
            optimizer.step()

            return

        experience = Experience(state, action, reward, next_state, non_terminal)
        self.replay_memory.add_experience(experience)

        if self._episode % self.replay_memory.update_after_episodes == 0:
            batch = self.replay_memory.sample_batch()

            # note: PyTorch's arange() works like range(), it is used in this case to select all q-values
            # and select the q-value for the taken action with batch.actions
            q = self.get_q_values(batch.states)[arange(len(batch.states)), batch.actions]
            # OPTIMIZATION NOTE: the future rewards are calculated even from non-terminal states
            target_q = batch.rewards + self.gamma * self.get_max_q_value(batch.next_states) * batch.non_terminal

            optimizer = self.network.get_optimizer()

            optimizer.zero_grad()
            loss = mse_loss(q, target_q)
            loss.backward()
            optimizer.step()
