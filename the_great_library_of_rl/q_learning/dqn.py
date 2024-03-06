from torch import Tensor, argmax, tensor
from torch import max as torch_max
from torch.nn.functional import mse_loss
from torch import float as torch_float

from the_great_library_of_rl.neural_network import NeuralNetwork
from the_great_library_of_rl.q_learning import QAgent


class DQN(QAgent):
    def __init__(self, network: NeuralNetwork, gamma: float):
        self.network = network
        self.gamma = gamma

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

    def get_max_q_value(self, state) -> float:
        """
        Get the maximum q-value for a given state.

        Args:
            state: State of the environment.

        Returns:
            float: Q-value of the best action.
        """

        return torch_max(self.get_q_values(state)).item()

    def update_q_value(self, state, action: int, reward: float, next_state):
        q = self.get_q_values(state)[action]
        target_q = tensor(reward + self.gamma * self.get_max_q_value(next_state), dtype=torch_float)

        optimizer = self.network.get_optimizer()

        optimizer.zero_grad()
        loss = mse_loss(q, target_q)
        loss.backward()
        optimizer.step()
