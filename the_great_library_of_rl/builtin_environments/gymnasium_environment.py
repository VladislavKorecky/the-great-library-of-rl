from gymnasium import make

from the_great_library_of_rl.environment import Environment


class GymnasiumEnvironment(Environment):
    """
    Built-in class for handling Gymnasium environments (https://gymnasium.farama.org/).

    Args:
        env_name (str): Name of the environment to create.
    """

    def __init__(self, env_name: str) -> None:
        self.env = make(env_name)

        self.num_of_actions = self.env.action_space.n
        self.env_name = env_name

        self.state, _ = self.env.reset()
        self.last_reward = 0
        self.terminated = False
        self.truncated = False

    def reset(self) -> None:
        self.state, _ = self.env.reset()
        self.last_reward = 0
        self.terminated = False
        self.truncated = False

    def get_state(self):
        return self.state

    def get_action_count(self) -> int:
        return self.num_of_actions

    def step(self, action: int) -> None:
        self.state, reward, self.terminated, self.truncated, _ = self.env.step(action)
        self.last_reward = float(reward)

    def get_reward(self) -> float:
        return self.last_reward

    def is_terminated(self) -> bool:
        return self.terminated or self.truncated

    def close(self) -> None:
        self.env.close()

    def set_evaluation(self, value: bool) -> None:
        # set the env to evaluation
        if value:
            self.env = make(self.env_name, render_mode="human")
            self.reset()
            return

        # set the env for training
        self.env = make(self.env_name)
        self.reset()
