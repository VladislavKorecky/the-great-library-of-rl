from the_great_library_of_rl.builtin_environments.gymnasium_environment import GymnasiumEnvironment
from the_great_library_of_rl.exploration_strategies.epsilon_greedy_strategy import EpsilonGreedyStrategy
from the_great_library_of_rl.q_learning.q_table import QTable
from the_great_library_of_rl.tester import Tester
from the_great_library_of_rl.trainer import Trainer


# CONFIG
EPOCHS = 10

LEARNING_RATE = 0.1
GAMMA = 0.95

EPSILON_START = 1
EPSILON_END = 0.05
EPSILON_DECAY = 0.01


# SETUP
env = GymnasiumEnvironment("CliffWalking-v0")
agent = QTable(env.get_action_count(), LEARNING_RATE, GAMMA)
exploration_strategy = EpsilonGreedyStrategy(EPSILON_START, EPSILON_END, EPSILON_DECAY)

trainer = Trainer(agent, env, exploration_strategy)
tester = Tester(agent, env)


# TRAINING
trainer.train(EPOCHS)
env.close()


# TESTING
tester.test()
