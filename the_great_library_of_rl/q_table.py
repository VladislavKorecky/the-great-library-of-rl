class QTable:
    """
    Q-Learning agent that works by storing the q-values in a table of states and actions.

    Args:
        num_of_actions (int): Number of possible actions in every state.
        learning_rate (float): Learning rate for updating the q-values.
        gamma (float): Decay rate for future rewards.
    """

    def __init__(self, num_of_actions: int, learning_rate: float, gamma: float):
        self.num_of_actions = num_of_actions
        self.learning_rate = learning_rate
        self.gamma = gamma

        # table to store the q-values
        self.table = {}

    def get_q_values(self, state) -> list[float]:
        """
        Return the q-values for a given state. Replace the q-values with zeros for non-registered states.

        Args:
            state: Information about the state of the environment.

        Returns:
            list[float]: Q-values for a given state.
        """

        q_values = self.table.get(state)

        # check if information about the state exists
        if q_values is None:
            return [0] * self.num_of_actions

        return q_values

    def get_action(self, state) -> int:
        """
        Return the best action given a state.

        Args:
            state: Information about the state of the environment.

        Returns:
            int: Best action according to the agent.
        """

        q_values = self.get_q_values(state)

        # index of the maximum q-value
        max_q_index = 0

        # find the index with the max value
        for i, value in enumerate(q_values):

            if value > q_values[max_q_index]:
                max_q_index = i

        return max_q_index

    def register_state(self, state, check_if_exists: bool = True):
        """
        Add a new state to the table.

        Args:
            state: Information about the state of the environment.
            check_if_exists (bool, optional): Check if the state already exists before editing the table. Default: True
        """

        if check_if_exists and self.state_exists(state):
            return

        self.table[state] = [0] * self.num_of_actions

    def state_exists(self, state) -> bool:
        """
        Check if the state is registered in the table.

        Args:
            state: Information about the state of the environment.

        Returns:
            bool: True if the state exists, False otherwise.
        """

        return self.table.get(state) is not None

    def get_q_value(self, state, action: int) -> float:
        """
        Return the q-value for a state/action pair.

        Args:
            state: Information about the state of the environment.
            action (int): Action to evaluate in the given state.

        Returns:
            float: Q-value of the state/action pair. 0 if the state is not registered in the table.
        """

        q_values = self.get_q_values(state)
        return q_values[action]

    def get_max_q_value(self, state) -> float:
        """
        Get the maximum q-value for a given state.

        Args:
            state: Information about the state of the environment.

        Returns:
            float: Q-value of the best action.
        """

        q_values = self.get_q_values(state)
        return max(q_values)

    def update_q_value(self, state, action: int, reward: float, next_state):
        """
        Update the q-value in the table.

        Args:
            state: Information about the state of the environment.
            action (int): Action taken in the state.
            reward (float): Reward experienced after taking the action.
            next_state: State reached after taking the action.
        """

        # make sure that the state is registered
        self.register_state(state)

        # decay the current q-value with the learning rate
        adjusted_current_value = self.get_q_value(state, action) * (1 - self.learning_rate)

        # calculate the expected future reward by looking at the q-value for the next state
        expected_future_reward = self.gamma * self.get_max_q_value(next_state)

        # calculate the target q-value by combining the current and future reward
        target_q = (reward + expected_future_reward) * self.learning_rate

        # update the table with the adjusted and target q-value
        self.table[state][action] = adjusted_current_value + target_q
