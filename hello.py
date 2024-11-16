from collections import defaultdict
import gymnasium as gym
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

class BlackjackAgent:
    def __init__(self, env: gym.Env, learning_rate: float, initial_epsilon: float, epsilon_decay: float, final_epsilon: float, discount_factor: float = .95):
        self.env = env

        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.initial_epsilon = initial_epsilon
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

        self.training_error = []

    def get_action(self, observation: tuple[int, int, bool]) -> int:
        # Sometimes randomly explore.
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        # Otherwise, choose the apparent best action.
        else:
            return int(np.argmax(self.q_values[observation]))

    def update(self, observation: tuple[int, int, bool], action: int, reward: float, terminated: bool, next_observation: tuple[int, int, bool]):
        future_q_value = np.max(self.q_values[next_observation])
        temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[observation][action]

        self.q_values[observation][action] += self.learning_rate * temporal_difference
        self.training_error.append(temporal_difference)
    
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)

# hyperparameters

learning_rate = .01
n_episodes = 100_0000
start_epsilon = 1
epsilon_decay = start_epsilon / (n_episodes / 2)
final_epsilon = .1

env = gym.make('Blackjack-v1', sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

agent = BlackjackAgent(env, learning_rate, start_epsilon, epsilon_decay, final_epsilon)

for episode in tqdm(range(n_episodes)):
    observation, info = env.reset()
    done = False

    while not done:
        action = agent.get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)

        agent.update(observation, action, reward, terminated, next_observation)

        done = terminated or truncated
        observation = next_observation
    
    agent.decay_epsilon()

env.close()

plt.plot(np.convolve(agent.training_error, np.ones(1000) / 1000, mode='valid'))
plt.show()