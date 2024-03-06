from torch import tensor
from torch import float as torch_float

from the_great_library_of_rl.builtin_environments.gymnasium_environment import GymnasiumEnvironment


class TensorGymnasiumEnvironment(GymnasiumEnvironment):
    """
    Gymnasium environment that returns states as Tensors. See: GymnasiumEnvironment

    Args:
        env_name (str): Name of the environment to create.
    """

    def __init__(self, env_name: str) -> None:
        super().__init__(env_name)

    def get_state(self):
        if hasattr(self.state, "__iter__"):
            return tensor(self.state, dtype=torch_float)

        return tensor([self.state], dtype=torch_float)
