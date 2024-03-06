from time import sleep

from the_great_library_of_rl.environment import Environment
from the_great_library_of_rl.q_learning import QAgent


class Tester:
    def __init__(self, agent: QAgent, environment: Environment) -> None:
        self.agent = agent
        self.environment = environment

    def test(self, delay: float = 0) -> None:
        """
        Test/Evaluate the agent in an environment.

        Args:
            delay (float, optional): Optional time delay between each step.
        """

        # make sure the env is in a testing/evaluation mode
        self.environment.set_evaluation(True)

        while True:
            state = self.environment.get_state()
            action = self.agent.get_action(state)
            self.environment.step(action)

            sleep(delay)

            if self.environment.is_terminated():
                break
