"""
Q-Learning Tabular Agent para LunarLander-v3
=============================================
Implementación clásica de Q-Learning con discretización del espacio de estados.
Sirve como baseline para demostrar las limitaciones del enfoque tabular
en espacios continuos de alta dimensionalidad.
"""

import numpy as np


class QLearningAgent:
    def __init__(
        self,
        n_actions: int = 4,
        n_bins: int = 8,
        lr: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
    ):
        self.n_actions = n_actions
        self.n_bins = n_bins
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Límites para discretización de cada dimensión del estado
        # LunarLander: [pos_x, pos_y, vel_x, vel_y, angle, angular_vel, leg_left, leg_right]
        self.state_bounds = [
            (-1.5, 1.5),    # pos_x
            (-0.5, 2.0),    # pos_y
            (-2.0, 2.0),    # vel_x
            (-2.0, 2.0),    # vel_y
            (-3.14, 3.14),  # angle
            (-5.0, 5.0),    # angular_vel
            (0, 1),         # leg_left (contacto)
            (0, 1),         # leg_right (contacto)
        ]

        # Crear bins para cada dimensión
        self.bins = [
            np.linspace(low, high, n_bins - 1)
            for low, high in self.state_bounds
        ]

        # Q-table: n_bins^8 estados × 4 acciones
        # Con n_bins=8: 8^8 = 16M entradas (grande pero manejable)
        self.q_table = {}

    def _discretize(self, state: np.ndarray) -> tuple:
        """Convierte estado continuo a índice discreto."""
        discrete = []
        for i, val in enumerate(state):
            idx = int(np.digitize(val, self.bins[i]))
            discrete.append(idx)
        return tuple(discrete)

    def _get_q_values(self, state_key: tuple) -> np.ndarray:
        """Obtiene Q-values para un estado, inicializa en 0 si no existe."""
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.n_actions)
        return self.q_table[state_key]

    def select_action(self, state: np.ndarray) -> int:
        """Selección epsilon-greedy."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)

        state_key = self._discretize(state)
        q_values = self._get_q_values(state_key)
        return int(np.argmax(q_values))

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Actualización Q-Learning: Q(s,a) += α[r + γ max Q(s',a') - Q(s,a)]"""
        s_key = self._discretize(state)
        ns_key = self._discretize(next_state)

        q_values = self._get_q_values(s_key)
        next_q_values = self._get_q_values(ns_key)

        target = reward + (1 - done) * self.gamma * np.max(next_q_values)
        q_values[action] += self.lr * (target - q_values[action])

    def decay_epsilon(self):
        """Decae epsilon después de cada episodio."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
