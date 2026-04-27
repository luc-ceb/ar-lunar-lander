"""
Deep Q-Network (DQN) Agent para LunarLander-v3
================================================
Implementación de DQN con los componentes clave:
- Red neuronal como aproximador de función Q
- Experience Replay Buffer
- Target Network con sincronización periódica
- Epsilon-greedy con decay
"""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    """Red neuronal que aproxima Q(s, a) para todas las acciones."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    """Buffer circular que almacena transiciones (s, a, r, s', done)."""

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 4,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        target_update_freq: int = 10,  # episodios entre sync de target network
        device: str = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Red principal (se entrena) y red target (se congela)
        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer = ReplayBuffer(buffer_size)
        self.episodes_done = 0

    def select_action(self, state: np.ndarray) -> int:
        """Selección epsilon-greedy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_t)
            return int(q_values.argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        """Almacena transición en el replay buffer."""
        self.buffer.push(state, action, reward, next_state, done)

    def update(self):
        """Entrena la red con un mini-batch del replay buffer."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(
            self.batch_size
        )

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # Q-values actuales: Q(s, a) para las acciones tomadas
        current_q = self.q_network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze()

        # Target: r + γ * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q = self.target_network(next_states_t).max(dim=1)[0]
            target_q = rewards_t + (1 - dones_t) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping para estabilidad
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    def end_episode(self):
        """Se llama al final de cada episodio: decay epsilon + sync target."""
        self.episodes_done += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        if self.episodes_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
