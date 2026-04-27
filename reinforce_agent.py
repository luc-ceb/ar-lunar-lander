"""
REINFORCE (Policy Gradient) Agent para LunarLander-v3
======================================================
Implementación del algoritmo REINFORCE con baseline para
reducción de varianza. Aprende directamente la política π(a|s)
en vez de estimar valores Q.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):
    """Red que outputea probabilidades π(a|s) sobre las acciones."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class REINFORCEAgent:
    def __init__(
        self,
        state_dim: int = 8,
        action_dim: int = 4,
        lr: float = 1e-3,
        gamma: float = 0.99,
        use_baseline: bool = True,
        device: str = None,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gamma = gamma
        self.use_baseline = use_baseline

        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Almacenamiento del episodio actual
        self.log_probs = []
        self.rewards = []

        # Baseline: promedio móvil de retornos
        self.baseline = 0.0
        self.baseline_alpha = 0.01  # tasa de actualización del baseline

    def select_action(self, state: np.ndarray) -> int:
        """Muestrea una acción de la distribución π(a|s)."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy(state_t)
        dist = Categorical(probs)
        action = dist.sample()

        self.log_probs.append(dist.log_prob(action))

        return action.item()

    def store_reward(self, reward: float):
        """Almacena la recompensa del paso actual."""
        self.rewards.append(reward)

    def update(self):
        """
        Actualiza la política al final del episodio.
        
        Gradiente: ∇J(θ) = Σ_t ∇log π(a_t|s_t) * (G_t - baseline)
        
        donde G_t = Σ_{k=t}^{T} γ^{k-t} * r_k  (retorno descontado desde t)
        """
        if not self.rewards:
            return 0.0

        # Calcular retornos descontados G_t para cada paso t
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns).to(self.device)

        # Normalizar retornos (reduce varianza)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Calcular loss: -log π(a|s) * (G - baseline)
        policy_loss = []
        for log_prob, G_t in zip(self.log_probs, returns):
            advantage = G_t - self.baseline if self.use_baseline else G_t
            policy_loss.append(-log_prob * advantage)

        loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Actualizar baseline con promedio móvil
        episode_return = sum(self.rewards)
        self.baseline += self.baseline_alpha * (episode_return - self.baseline)

        # Limpiar buffers del episodio
        loss_val = loss.item()
        self.log_probs = []
        self.rewards = []

        return loss_val
