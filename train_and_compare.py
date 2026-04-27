"""
Entrenamiento y comparación de agentes RL en LunarLander-v3
=============================================================
Entrena Q-Learning, DQN y REINFORCE, genera gráficas comparativas
y una tabla resumen de métricas.

Uso:
    python train_and_compare.py

Los resultados se guardan en la carpeta results/
"""

import os
import time
from typing import Dict, List

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from q_learning_agent import QLearningAgent
from dqn_agent import DQNAgent
from reinforce_agent import REINFORCEAgent

# ──────────────────────────────────────────────────────────────
# Configuración
# ──────────────────────────────────────────────────────────────
N_EPISODES = 1000         # Episodios de entrenamiento por agente
N_SEEDS = 3               # Seeds para promediar resultados
SOLVED_THRESHOLD = 200    # Reward promedio para considerar "resuelto"
WINDOW = 50               # Ventana de media móvil para suavizar curvas
RESULTS_DIR = "results"

os.makedirs(RESULTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
# Funciones de entrenamiento
# ──────────────────────────────────────────────────────────────

def train_q_learning(env, n_episodes: int, seed: int) -> List[float]:
    """Entrena Q-Learning tabular y retorna rewards por episodio."""
    np.random.seed(seed)
    agent = QLearningAgent(
        n_actions=env.action_space.n,
        n_bins=8,
        lr=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
    )
    rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.update(state, action, reward, next_state, done or truncated)
            state = next_state
            total_reward += reward

        agent.decay_epsilon()
        rewards.append(total_reward)

        if (ep + 1) % 100 == 0:
            avg = np.mean(rewards[-WINDOW:])
            print(f"  [Q-Learning] Ep {ep+1}/{n_episodes} | "
                  f"Reward avg({WINDOW}): {avg:.1f} | ε: {agent.epsilon:.3f}")

    return rewards


def train_dqn(env, n_episodes: int, seed: int) -> List[float]:
    """Entrena DQN y retorna rewards por episodio."""
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)

    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=64,
        buffer_size=100_000,
        target_update_freq=10,
    )
    rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done or truncated)
            agent.update()
            state = next_state
            total_reward += reward

        agent.end_episode()
        rewards.append(total_reward)

        if (ep + 1) % 100 == 0:
            avg = np.mean(rewards[-WINDOW:])
            print(f"  [DQN]        Ep {ep+1}/{n_episodes} | "
                  f"Reward avg({WINDOW}): {avg:.1f} | ε: {agent.epsilon:.3f}")

    return rewards


def train_reinforce(env, n_episodes: int, seed: int) -> List[float]:
    """Entrena REINFORCE y retorna rewards por episodio."""
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)

    agent = REINFORCEAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=1e-3,
        gamma=0.99,
        use_baseline=True,
    )
    rewards = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        total_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            action = agent.select_action(state)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.store_reward(reward)
            state = next_state
            total_reward += reward

        agent.update()
        rewards.append(total_reward)

        if (ep + 1) % 100 == 0:
            avg = np.mean(rewards[-WINDOW:])
            print(f"  [REINFORCE]  Ep {ep+1}/{n_episodes} | "
                  f"Reward avg({WINDOW}): {avg:.1f}")

    return rewards


# ──────────────────────────────────────────────────────────────
# Utilidades para análisis
# ──────────────────────────────────────────────────────────────

def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Media móvil para suavizar curvas de reward."""
    return np.convolve(data, np.ones(window) / window, mode="valid")


def episodes_to_solve(rewards: np.ndarray, threshold: float, window: int) -> int:
    """Primer episodio donde la media móvil supera el threshold."""
    ma = moving_average(rewards, window)
    solved = np.where(ma >= threshold)[0]
    if len(solved) > 0:
        return solved[0] + window
    return -1  # No convergió


def compute_metrics(all_rewards: Dict[str, List[List[float]]]) -> Dict:
    """Calcula métricas resumen para cada agente."""
    metrics = {}
    for name, seed_rewards in all_rewards.items():
        # Promediar últimos 100 episodios de cada seed
        final_rewards = [np.mean(r[-100:]) for r in seed_rewards]
        solve_eps = [
            episodes_to_solve(np.array(r), SOLVED_THRESHOLD, WINDOW)
            for r in seed_rewards
        ]
        solve_eps_valid = [e for e in solve_eps if e > 0]

        metrics[name] = {
            "reward_mean": np.mean(final_rewards),
            "reward_std": np.std(final_rewards),
            "episodes_to_solve": (
                np.mean(solve_eps_valid) if solve_eps_valid else float("inf")
            ),
            "solved_ratio": len(solve_eps_valid) / len(solve_eps),
        }
    return metrics


# ──────────────────────────────────────────────────────────────
# Gráficas
# ──────────────────────────────────────────────────────────────

def plot_comparison(all_rewards: Dict[str, List[List[float]]], metrics: Dict):
    """Genera la gráfica comparativa principal."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"Q-Learning": "#e74c3c", "DQN": "#2ecc71", "REINFORCE": "#3498db"}

    # --- Panel 1: Curvas de aprendizaje ---
    ax1 = axes[0]
    for name, seed_rewards in all_rewards.items():
        # Calcular media y banda de confianza entre seeds
        min_len = min(len(r) for r in seed_rewards)
        aligned = np.array([r[:min_len] for r in seed_rewards])

        # Media móvil por seed, luego promediar
        smoothed = np.array(
            [moving_average(r, WINDOW) for r in aligned]
        )
        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)

        x = np.arange(len(mean)) + WINDOW
        ax1.plot(x, mean, label=name, color=colors[name], linewidth=2)
        ax1.fill_between(x, mean - std, mean + std, color=colors[name], alpha=0.15)

    ax1.axhline(y=SOLVED_THRESHOLD, color="gray", linestyle="--", alpha=0.5,
                label=f"Umbral resuelto ({SOLVED_THRESHOLD})")
    ax1.set_xlabel("Episodio")
    ax1.set_ylabel("Reward (media móvil)")
    ax1.set_title("Curvas de Aprendizaje")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Tabla de métricas ---
    ax2 = axes[1]
    ax2.axis("off")
    table_data = []
    for name in all_rewards.keys():
        m = metrics[name]
        eps_str = (
            f"{m['episodes_to_solve']:.0f}" if m["episodes_to_solve"] < float("inf")
            else "No convergió"
        )
        table_data.append([
            name,
            f"{m['reward_mean']:.1f} ± {m['reward_std']:.1f}",
            eps_str,
            f"{m['solved_ratio']*100:.0f}%",
        ])

    table = ax2.table(
        cellText=table_data,
        colLabels=["Agente", "Reward Final", "Eps. Convergencia", "% Resuelto"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    ax2.set_title("Métricas Comparativas", fontsize=13, pad=20)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "comparacion_agentes.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nGráfica guardada en: {path}")


def plot_individual_curves(all_rewards: Dict[str, List[List[float]]]):
    """Genera gráficas individuales por agente con todas las seeds."""
    colors = {"Q-Learning": "#e74c3c", "DQN": "#2ecc71", "REINFORCE": "#3498db"}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)

    for ax, (name, seed_rewards) in zip(axes, all_rewards.items()):
        for i, rewards in enumerate(seed_rewards):
            ma = moving_average(np.array(rewards), WINDOW)
            x = np.arange(len(ma)) + WINDOW
            ax.plot(x, ma, alpha=0.5, label=f"Seed {i+1}", linewidth=1)

        ax.axhline(y=SOLVED_THRESHOLD, color="gray", linestyle="--", alpha=0.5)
        ax.set_title(name, color=colors[name], fontweight="bold")
        ax.set_xlabel("Episodio")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Reward (media móvil)")
    plt.suptitle("Estabilidad entre seeds", fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "estabilidad_seeds.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfica guardada en: {path}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  RL Comparativo: Q-Learning vs DQN vs REINFORCE")
    print("  Entorno: LunarLander-v3")
    print(f"  Episodios: {N_EPISODES} | Seeds: {N_SEEDS}")
    print("=" * 60)

    trainers = {
        "Q-Learning": train_q_learning,
        "DQN": train_dqn,
        "REINFORCE": train_reinforce,
    }

    all_rewards: Dict[str, List[List[float]]] = {name: [] for name in trainers}
    training_times: Dict[str, List[float]] = {name: [] for name in trainers}

    for name, train_fn in trainers.items():
        for seed in range(N_SEEDS):
            print(f"\n{'─'*50}")
            print(f"Entrenando {name} | Seed {seed+1}/{N_SEEDS}")
            print(f"{'─'*50}")

            env = gym.make("LunarLander-v3")
            start_time = time.time()
            rewards = train_fn(env, N_EPISODES, seed=seed * 42)
            elapsed = time.time() - start_time
            env.close()

            all_rewards[name].append(rewards)
            training_times[name].append(elapsed)
            print(f"  Tiempo: {elapsed:.1f}s | "
                  f"Reward final (últ. 100): {np.mean(rewards[-100:]):.1f}")

    # Calcular métricas
    metrics = compute_metrics(all_rewards)

    # Agregar tiempos de entrenamiento
    for name in metrics:
        metrics[name]["train_time_mean"] = np.mean(training_times[name])

    # Imprimir resumen
    print(f"\n{'='*60}")
    print("RESUMEN DE RESULTADOS")
    print(f"{'='*60}")
    for name, m in metrics.items():
        print(f"\n{name}:")
        print(f"  Reward final:          {m['reward_mean']:.1f} ± {m['reward_std']:.1f}")
        eps = m["episodes_to_solve"]
        print(f"  Episodios convergencia: {eps:.0f}" if eps < float("inf")
              else "  Episodios convergencia: No convergió")
        print(f"  Tasa resolución:       {m['solved_ratio']*100:.0f}%")
        print(f"  Tiempo entrenamiento:  {m['train_time_mean']:.1f}s")

    # Generar gráficas
    print(f"\n{'─'*50}")
    print("Generando gráficas...")
    plot_comparison(all_rewards, metrics)
    plot_individual_curves(all_rewards)

    # Guardar rewards crudos para análisis posterior
    """
    np.savez(
        os.path.join(RESULTS_DIR, "rewards_raw.npz"),
        **{f"{name}_seed{i}": np.array(r)
           for name, seeds in all_rewards.items()
           for i, r in enumerate(seeds)},
    )
    print(f"Datos crudos guardados en: {RESULTS_DIR}/rewards_raw.npz")
    """
    print("\nEjecutado correctamente")


if __name__ == "__main__":
    main()
