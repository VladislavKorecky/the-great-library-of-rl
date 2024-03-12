from torch import Tensor
from torch.nn import Sequential, Linear, LeakyReLU
from torch.optim import Adam, Optimizer

from the_great_library_of_rl.builtin_environments.tensor_gymnasium_environment import TensorGymnasiumEnvironment
from the_great_library_of_rl.exploration_strategies.epsilon_greedy_strategy import EpsilonGreedyStrategy
from the_great_library_of_rl.neural_network import NeuralNetwork
from the_great_library_of_rl.q_learning.dqn import DQN
from the_great_library_of_rl.q_learning.replay_memory import ReplayMemory
from the_great_library_of_rl.tester import Tester
from the_great_library_of_rl.trainer import Trainer


# CONFIG
EPOCHS = 10000

LEARNING_RATE = 0.001
GAMMA = 0.95

EPSILON_START = 1
EPSILON_END = 0.05
EPSILON_DECAY = 1 / 10000


# NETWORK
class Model(NeuralNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers = Sequential(
            Linear(4, 5),
            LeakyReLU(),

            Linear(5, 5),
            LeakyReLU(),

            Linear(5, 2)
        )
        self.optimizer = None

    def get_optimizer(self) -> Optimizer:
        if self.optimizer is None:
            self.optimizer = Adam(self.parameters(), lr=LEARNING_RATE)

        return self.optimizer

    def forward(self, x: Tensor) -> Tensor:
        return self.layers.forward(x)


# SETUP
model = Model()
replay_memory = ReplayMemory(100000, 30, 64)
env = TensorGymnasiumEnvironment("CartPole-v1")
agent = DQN(model, GAMMA, replay_memory=replay_memory)
exploration_strategy = EpsilonGreedyStrategy(EPSILON_START, EPSILON_END, EPSILON_DECAY)

trainer = Trainer(agent, env, exploration_strategy)
tester = Tester(agent, env)


# TRAINING
trainer.train(EPOCHS)
env.close()


# TESTING
tester.test()
