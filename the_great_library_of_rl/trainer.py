from random import randrange

from the_great_library_of_rl.environment import Environment
from the_great_library_of_rl.exploration_strategies.epsilon_greedy_strategy import EpsilonGreedyStrategy
from the_great_library_of_rl.q_table import QTable


class Trainer:
    def __init__(self, agent: QTable, environment: Environment, exploration_strategy: EpsilonGreedyStrategy):
        self.agent = agent
        self.environment = environment
        self.exploration_strategy = exploration_strategy

    def train(self, epochs: int) -> None:
        """
        Train the agent on the given environment.

        Args:
            epochs: Number of times the agent should train in the environment.
        """

        # make sure the env is in a training phase
        self.environment.set_evaluation(False)

        for _ in range(epochs):
            self.execute_epoch()

            # reset the environment
            self.environment.reset()

            # update epsilon
            self.exploration_strategy.decay_epsilon()

    def execute_epoch(self) -> None:
        """
        Execute one epoch of training.
        """

        run = True

        while run:
            # get the current state of the env
            state = self.environment.get_state()

            # choose an action with respect to the exploration strategy
            action = self.__get_action(state)

            # execute the action
            self.environment.step(action)

            # pull important information from the environment
            reward = self.environment.get_reward()
            next_state = self.environment.get_state()

            # update the agent's brain
            self.agent.update_q_value(state, action, reward, next_state)

            # check if the loop should keep going
            run = not self.environment.is_terminated()

    def __get_action(self, state) -> int:
        """
        Choose an action with respect to the exploration strategy.

        Args:
            state: Information about the state of the environment.

        Returns:
            int: The action to take. Either random or determined by the agent.
        """

        if self.exploration_strategy.should_sample_random_action():
            return randrange(self.environment.get_action_count())  # pick a random action

        return self.agent.get_action(state)  # pick agent's action
