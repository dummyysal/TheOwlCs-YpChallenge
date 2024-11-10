import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

class CyberAttackEnv(gym.Env):
    def __init__(self, dataset, max_steps_per_episode=200):
        super(CyberAttackEnv, self).__init__()

        self.dataset = dataset
        self.current_index = 0
        self.max_steps_per_episode = max_steps_per_episode
        self.steps_taken = 0

        n_features = len(self.dataset.columns) - 2  # Adjusted for excluded columns
        self.observation_space = spaces.Box(low=0, high=1, shape=(n_features,), dtype=np.float32)

        self.action_space = spaces.Discrete(5)

    def reset(self):
        self.current_index = np.random.randint(0, len(self.dataset))
        self.steps_taken = 0
        return self._get_state()

    def _get_state(self):
        state = self.dataset.iloc[self.current_index].drop(['attack_cat', 'label']).values
        return np.array(state, dtype=np.float32)

    def step(self, action):
        row = self.dataset.iloc[self.current_index]
        actual_attack = row['attack_cat']

        reward = self._calculate_reward(action, actual_attack)

        self.current_index = (self.current_index + 1) % len(self.dataset)
        self.steps_taken += 1

        done = (self.steps_taken >= self.max_steps_per_episode) or (self.current_index == 0)
        return self._get_state(), reward, done, {}

    def _calculate_reward(self, action, actual_attack):
        if actual_attack == 'Normal':
            if action == 0:
                return 1
            else:
                return -1
        else:
            if action == 1 and actual_attack in ['Backdoor', 'Exploits', 'Worms']:
                return 2
            elif action == 2 and actual_attack in ['Analysis', 'Reconnaissance']:
                return 2
            elif action == 3 and actual_attack == 'Normal':
                return 1
            else:
                return -2




class DQNModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self, state_size, action_size, buffer_size=100000, batch_size=64, gamma=0.95, tau=0.005,
                 learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = deque(maxlen=buffer_size)

        # Initialize Q-Network and target Q-Network
        self.model = DQNModel(state_size, action_size)
        self.target_model = DQNModel(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.update_target_model()

    def update_target_model(self):
        # Soft update of target network parameters using tau
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, epsilon=0.1):
        if np.random.rand() <= epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()

    def step(self, state, action, reward, next_state, done):
        # Save the experience to memory
        self.remember(state, action, reward, next_state, done)

        # Replay if enough samples in memory
        if len(self.memory) >= self.batch_size:
            self.replay()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in batch:
            state_tensor = torch.FloatTensor(state)
            next_state_tensor = torch.FloatTensor(next_state)

            # Get current Q-values for action selection
            current_q_values = self.model(state_tensor).squeeze()
            target_q_values = current_q_values.clone().detach()

            target = reward
            if not done:
                with torch.no_grad():
                    next_state_q_values = self.target_model(next_state_tensor).squeeze()
                    target += self.gamma * torch.max(next_state_q_values).item()

            target_q_values[action] = target
            loss = nn.functional.mse_loss(current_q_values[action], torch.tensor(target).float())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay