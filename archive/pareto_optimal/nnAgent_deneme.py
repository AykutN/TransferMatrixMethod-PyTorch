import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np
import environment_deneme as env
import torch.nn.functional as F

class MultiObjectiveQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(MultiObjectiveQNetwork, self).__init__()
        # Ortak katmanlar
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        # AVT ve JPH için ayrı çıkışlar
        self.head_avt = nn.Linear(hidden_dim, action_dim)
        self.head_jph = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, objective='both'):
        features = self.shared(state)
        if objective == 'AVT':
            return self.head_avt(features)
        elif objective == 'JPH':
            return self.head_jph(features)
        else:
            q_avt = self.head_avt(features)
            q_jph = self.head_jph(features)
            return q_avt, q_jph

class ReplayBuffer:
    def __init__(self, max_size, state_dim, alpha=0.6):
        self.max_size = max_size
        self.state_dim = state_dim
        self.memory = []
        self.position = 0
        self.priorities = np.zeros((max_size,), dtype=np.float32)
        self.alpha = alpha

    def store(self, state, action, reward_vector, next_state, done):
        state = np.array(state) if not isinstance(state, np.ndarray) else state
        next_state = np.array(next_state) if not isinstance(next_state, np.ndarray) else state
        
        
        reward_avt, reward_jph = reward_vector
        
        priority = abs(reward_avt) + abs(reward_jph)
        
        max_priority = self.priorities.max() if self.memory else 1.0

        if len(self.memory) < self.max_size:
            self.memory.append(None)

        self.memory[self.position] = (state, action, reward_vector, next_state, done)
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

        states, actions, reward_vectors, next_states, dones = zip(*samples)
        reward_avt, reward_jph = zip(*reward_vectors)

        return (np.array(states), np.array(actions), np.array(reward_avt), np.array(reward_jph),
                np.array(next_states), np.array(dones), indices, weights)

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

        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.q_network = MultiObjectiveQNetwork(state_dim, action_dim).to(self.device)
        self.target_network = MultiObjectiveQNetwork(state_dim, action_dim).to(self.device)
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
            state = T.tensor(state, dtype=T.float32).reshape(-1).to(self.device)
            q_avt, q_jph = self.q_network(state)
            # AVT ≥ 25 koşulunu önceliklendir
            q_values = T.stack([q_avt, q_jph], dim=1)  # [action_dim, 2]
            dominant_actions = self.find_pareto_dominant(q_values)
            return np.random.choice(dominant_actions)  # Dominant eylemlerden rastgele birini seç

    def find_pareto_dominant(self, q_values):
        dominant = []
        for i in range(self.action_dim):
            is_dominated = False
            for j in range(self.action_dim):
                if i != j and (q_values[j][0] >= q_values[i][0] and q_values[j][1] > q_values[i][1]) or \
                              (q_values[j][0] > q_values[i][0] and q_values[j][1] >= q_values[i][1]):
                    is_dominated = True
                    break
            if not is_dominated:
                dominant.append(i)
        if not dominant:
            return [T.argmax(q_values[:, 0] + q_values[:, 1]).item()]  # Fallback
        return dominant

    def train(self):
        if len(self.buffer.memory) < self.batch_size:
            return None

        states, actions, rewards_avt, rewards_jph, next_states, dones, indices, weights = self.buffer.sample(self.batch_size)
        states = np.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        next_states = np.nan_to_num(next_states, nan=0.0, posinf=0.0, neginf=0.0)

        states = T.tensor(states, dtype=T.float32).reshape(self.batch_size, -1).to(self.device)
        next_states = T.tensor(next_states, dtype=T.float32).reshape(self.batch_size, -1).to(self.device)
        actions = T.tensor(actions, dtype=T.int64).unsqueeze(1).to(self.device)
        rewards_avt = T.tensor(rewards_avt, dtype=T.float32).unsqueeze(1).to(self.device)
        rewards_jph = T.tensor(rewards_jph, dtype=T.float32).unsqueeze(1).to(self.device)
        dones = T.tensor(dones, dtype=T.float32).unsqueeze(1).to(self.device)
        weights = T.tensor(weights, dtype=T.float32).unsqueeze(1).to(self.device)

        q_avt, q_jph = self.q_network(states)
        q_avt = q_avt.gather(1, actions)
        q_jph = q_jph.gather(1, actions)

        with T.no_grad():
            next_q_avt, next_q_jph = self.target_network(next_states)
            next_q_avt = next_q_avt.max(1, keepdim=True)[0]
            next_q_jph = next_q_jph.max(1, keepdim=True)[0]
            targets_avt = rewards_avt + self.gamma * next_q_avt * (1 - dones)
            targets_jph = rewards_jph + self.gamma * next_q_jph * (1 - dones)

        loss_avt = (self.criterion(q_avt, targets_avt) * weights).mean()
        loss_jph = (self.criterion(q_jph, targets_jph) * weights).mean()
        total_loss = loss_avt + loss_jph

        if T.isnan(total_loss) or T.isinf(total_loss):
            print("NaN or inf detected in loss")
            return None

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        self.step += 1

        td_errors_avt = (q_avt - targets_avt).detach().cpu().numpy()
        td_errors_jph = (q_jph - targets_jph).detach().cpu().numpy()
        priorities = np.abs(td_errors_avt) + np.abs(td_errors_jph) + 1e-5
        self.buffer.update_priorities(indices, priorities)

        return total_loss.item()