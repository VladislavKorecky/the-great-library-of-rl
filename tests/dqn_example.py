from torch import Tensor
from torch.nn import Sequential, Linear, LeakyReLU
from torch.optim import Adam, Optimizer

from the_great_library_of_rl.builtin_environments.tensor_gymnasium_environment import TensorGymnasiumEnvironment
from the_great_library_of_rl.exploration_strategies.epsilon_greedy_strategy import EpsilonGreedyStrategy
from the_great_library_of_rl.neural_network import NeuralNetwork
from the_great_library_of_rl.q_learning.dqn import DQN
from the_great_library_of_rl.tester import Tester
from the_great_library_of_rl.trainer import Trainer


# CONFIG
EPOCHS = 20

LEARNING_RATE = 0.001
GAMMA = 0.95

EPSILON_START = 1
EPSILON_END = 0.05
EPSILON_DECAY = 0.005


# NETWORK
class Model(NeuralNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.layers = Sequential(
            Linear(1, 10),
            LeakyReLU(),

            Linear(10, 10),
            LeakyReLU(),

            Linear(10, 4)
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
env = TensorGymnasiumEnvironment("CliffWalking-v0")
agent = DQN(model, GAMMA)
exploration_strategy = EpsilonGreedyStrategy(EPSILON_START, EPSILON_END, EPSILON_DECAY)

trainer = Trainer(agent, env, exploration_strategy)
tester = Tester(agent, env)


# TRAINING
trainer.train(EPOCHS)
env.close()


# TESTING
tester.test()
