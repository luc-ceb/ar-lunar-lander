# TP Aprendizaje por Refuerzo — LunarLander-v2

Comparación de tres algoritmos de RL aplicados al problema de aterrizaje lunar:

1. **Q-Learning tabular** (baseline con discretización)
2. **Deep Q-Network (DQN)** (aproximación funcional con replay + target network)
3. **REINFORCE** (policy gradient con baseline)

## Setup

```bash
# Crear entorno virtual (opcional pero recomendado)
python -m venv venv
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

> **Nota:** `gymnasium[box2d]` requiere `swig`. En Ubuntu:
> ```bash
> sudo apt install swig
> ```

## Ejecución

```bash
python train_and_compare.py
```

El script entrena los 3 agentes con 3 seeds diferentes (configurable) y genera:

- `results/comparacion_agentes.png` — Curvas de aprendizaje + tabla de métricas
- `results/estabilidad_seeds.png` — Variabilidad entre seeds por agente
- `results/rewards_raw.npz` — Datos crudos para análisis adicional

## Estructura

```
├── q_learning_agent.py      # Q-Learning tabular
├── dqn_agent.py              # DQN con replay buffer + target network
├── reinforce_agent.py        # REINFORCE con baseline
├── train_and_compare.py      # Script principal de entrenamiento y comparación
├── requirements.txt
└── README.md
```

## Configuración

`train_and_compare.py` con parametros ajustables:

- `N_EPISODES`: episodios por agente (default: 1000)
- `N_SEEDS`: seeds para promediar (default: 3)
- `SOLVED_THRESHOLD`: reward para "resolver" el env (default: 200)
- `WINDOW`: ventana de media móvil (default: 50)

## Resultados esperados

- **Q-Learning**: reward final ~-100 a 0. No resuelve el problema por la discretización.
- **DQN**: reward final ~200+. Resuelve en ~400-600 episodios.
- **REINFORCE**: reward final ~100-200. Más lento y con mayor varianza que DQN.
