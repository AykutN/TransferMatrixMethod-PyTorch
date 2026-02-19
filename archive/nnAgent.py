import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import environment as env

import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=0.1)  
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = self.dropout1(x)  
        x = F.relu(self.linear2(x))
        x = self.dropout2(x)  
        q_values = self.linear3(x)
        return q_values



class ReplayBuffer():
    def __init__(self, max_size, state_dim, alpha=0.6):
        self.max_size = max_size
        self.state_dim = state_dim
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((max_size,), dtype=np.float32)
        self.alpha = alpha
        self.env = env.Env()

    def store(self, state, action, reward, next_state, done, avt_value, highest_avt, jph_value, highest_jph):
        state = np.array(state) if not isinstance(state, np.ndarray) else state
        next_state = np.array(next_state) if not isinstance(next_state, np.ndarray) else next_state

        self.highest_avt = highest_avt
        self.highest_jph = highest_jph
        
        
        bounds = {
            "d1": (5, 70),
            "d2": (5, 25),
            "d3": (10, 50),
            "d4": (50, 150),
            "d5": (10, 50),
            "d6": (5, 25),
            "d7": (5, 70), 
        }

        valid = all(bounds[key][0] <= getattr(self.env, f"{key}p") <= bounds[key][1] for key in bounds)
        priority=0
        if valid:
            if avt_value > 25 and jph_value > self.highest_jph:
                priority += 20
            if avt_value > 25:
                priority += 10
        else:
            priority -= 10


        max_priority = self.priorities.max() if self.memory else 1.0

        if len(self.memory) < self.max_size:
            self.memory.append(None)
            

        self.memory[self.position] = (state, action, reward, next_state, done, priority)
        self.priorities[self.position] = max(priority, max_priority)
        self.position = (self.position + 1) % self.max_size

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.max_size:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states, actions, rewards, next_states, dones, priorities = zip(*samples)

        return (np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones), indices, weights)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=0.005, gamma=0.99, buffer_size=10000, batch_size=128):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.epsilon = 1.0
        self.batch_size = batch_size

        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.step = 0  

        self.target_network.eval()
        self.update_target_network()
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7500, gamma=0.95)
        self.criterion = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_size, state_dim)

        print("DQNAgent initialized")

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_dim)
        else:
            state = T.tensor(state, dtype=T.float32).reshape(-1)
            q_values = self.q_network(state)
            return T.argmax(q_values, dim=0).item()

    def train(self):
        if len(self.buffer.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, indices, weights = self.buffer.sample(self.batch_size)
        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        next_states = np.nan_to_num(next_states, nan=0.0, posinf=0.0, neginf=0.0)

        if states is None:
            print("Sampling failed due to NaN values")
            return None

        states = T.tensor(states, dtype=T.float32).reshape(self.batch_size, -1)
        next_states = T.tensor(next_states, dtype=T.float32).reshape(self.batch_size, -1)
        actions = T.tensor(actions, dtype=T.int64).unsqueeze(1)
        rewards = T.tensor(rewards, dtype=T.float32).unsqueeze(1)
        dones = T.tensor(dones, dtype=T.float32).unsqueeze(1)
        weights = T.tensor(weights, dtype=T.float32).unsqueeze(1)

        q_values = self.q_network(states).gather(1, actions)
        with T.no_grad():
            next_q_values = self.target_network(next_states).max(1, keepdim=True)[0]
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        if T.isnan(q_values).any() or T.isnan(targets).any() or T.isinf(q_values).any() or T.isinf(targets).any():
            print("NaN or inf detected in q_values or targets")
            return None

        loss = (self.criterion(q_values, targets) * weights).mean()

        if T.isnan(loss) or T.isinf(loss):
            print("NaN or inf detected in loss")
            return None

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.step += 1

        # Update priorities
        td_errors = (q_values - targets).detach().cpu().numpy()
        priorities = np.abs(td_errors) + 1e-5
        self.buffer.update_priorities(indices, priorities)

        return loss.item()