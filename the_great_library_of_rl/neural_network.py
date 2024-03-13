from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer


class NeuralNetwork(Module):
    def forward(self, x: Tensor) -> Tensor:
        """
        Return the network's output after a feed forward pass.

        Args:
            x (Tensor): Network's input.

        Returns:
            Tensor: Network's output.
        """

        raise NotImplementedError

    def get_optimizer(self) -> Optimizer:
        """
        Return an optimizer used for this model.

        Returns:
            Optimizer: Optimizer used for this model.
        """

        raise NotImplementedError
