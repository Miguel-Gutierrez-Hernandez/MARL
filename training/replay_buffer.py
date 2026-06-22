"""
training/replay_buffer.py
==========================
Replay buffers para los algoritmos de RL.

¿Por qué hace falta un replay buffer?
  Los algoritmos off-policy (IQL, QMIX, MADDPG) no aprenden de cada
  experiencia en el momento en que ocurre, sino que la almacenan y
  muestrean mini-batches aleatorios para entrenamiento.
  Esto rompe la correlación temporal entre muestras consecutivas,
  lo que estabiliza el aprendizaje.

  Los algoritmos on-policy (MAPPO) usan un RolloutBuffer que almacena
  una trayectoria completa y la descarta después de usarla.

Hay dos clases:
  - MultiAgentReplayBuffer : para IQL, QMIX, MADDPG (off-policy)
  - RolloutBuffer          : para MAPPO (on-policy)
"""

from __future__ import annotations

import numpy as np
import torch
from collections import deque
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# Off-policy buffer (IQL, QMIX, MADDPG)
# ══════════════════════════════════════════════════════════════════════════════

class MultiAgentReplayBuffer:
    """
    Experience Replay para entornos multi-agente.

    Almacena transiciones (s, a, r, s', done) de TODOS los agentes juntos,
    más el estado global (para algoritmos CTDE como QMIX y MADDPG).

    Parámetros
    ----------
    capacity     : máximo de transiciones almacenadas (las antiguas se borran)
    agent_ids    : lista de identificadores de agente
    obs_shapes   : dict[agent_id → int] tamaño de la observación de cada agente
    n_actions    : dict[agent_id → int] número de acciones de cada agente
    state_dim    : dimensión del estado global (para CTDE); 0 = no se guarda
    """

    def __init__(
        self,
        capacity:   int,
        agent_ids:  list[str],
        obs_shapes: dict[str, int],
        n_actions:  dict[str, int],
        state_dim:  int = 0,
    ):
        self.capacity  = capacity
        self.agents    = agent_ids
        self.state_dim = state_dim
        self._ptr      = 0       # puntero circular
        self._size     = 0       # elementos almacenados actualmente

        # Pre-alocar arrays NumPy (más rápido que listas de Python)
        self._obs      = {a: np.zeros((capacity, obs_shapes[a]), dtype=np.float32)
                          for a in agent_ids}
        self._next_obs = {a: np.zeros((capacity, obs_shapes[a]), dtype=np.float32)
                          for a in agent_ids}
        self._actions  = {a: np.zeros((capacity,), dtype=np.int64)
                          for a in agent_ids}
        self._rewards  = {a: np.zeros((capacity,), dtype=np.float32)
                          for a in agent_ids}
        self._done     = np.zeros((capacity,), dtype=np.float32)

        if state_dim > 0:
            self._state      = np.zeros((capacity, state_dim), dtype=np.float32)
            self._next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        else:
            self._state = self._next_state = None

    def store(
        self,
        obs:        dict[str, np.ndarray],
        actions:    dict[str, int],
        rewards:    dict[str, float],
        next_obs:   dict[str, np.ndarray],
        done:       bool,
        state:      Optional[np.ndarray] = None,
        next_state: Optional[np.ndarray] = None,
    ) -> None:
        """Guarda una transición en el buffer (circular)."""
        idx = self._ptr

        for a in self.agents:
            self._obs     [a][idx] = obs     [a]
            self._next_obs[a][idx] = next_obs[a]
            self._actions [a][idx] = actions [a]
            self._rewards [a][idx] = rewards [a]

        self._done[idx] = float(done)

        if self._state is not None and state is not None:
            self._state     [idx] = state
            self._next_state[idx] = next_state

        self._ptr  = (idx + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: str = "cpu") -> dict:
        """
        Muestrea un mini-batch aleatorio y lo devuelve como tensores PyTorch.

        Returns
        -------
        batch : dict con claves "obs", "actions", "rewards", "next_obs",
                "done", y opcionalmente "state" / "next_state".
        """
        assert self._size >= batch_size, (
            f"Buffer con {self._size} elementos, necesitas {batch_size} para muestrear."
        )
        idxs = np.random.randint(0, self._size, size=batch_size)

        def to_t(arr):
            return torch.tensor(arr[idxs], dtype=torch.float32).to(device)

        batch = {
            "obs":      {a: to_t(self._obs[a])      for a in self.agents},
            "next_obs": {a: to_t(self._next_obs[a]) for a in self.agents},
            "actions":  {a: torch.tensor(self._actions[a][idxs],
                                          dtype=torch.long).to(device)
                         for a in self.agents},
            "rewards":  {a: to_t(self._rewards[a]) for a in self.agents},
            "done":     torch.tensor(self._done[idxs],
                                     dtype=torch.float32).to(device),
        }

        if self._state is not None:
            batch["state"]      = to_t(self._state)
            batch["next_state"] = to_t(self._next_state)

        return batch

    def __len__(self) -> int:
        return self._size

    def ready(self, batch_size: int) -> bool:
        """True si hay suficientes muestras para entrenar."""
        return self._size >= batch_size


# ══════════════════════════════════════════════════════════════════════════════
# On-policy buffer (MAPPO)
# ══════════════════════════════════════════════════════════════════════════════

class RolloutBuffer:
    """
    Buffer de trayectorias para algoritmos on-policy (MAPPO).

    A diferencia del replay buffer, este almacena una trayectoria completa,
    calcula las ventajas (GAE) y se vacía después de cada actualización.

    Parámetros
    ----------
    rollout_length : pasos de simulación antes de cada actualización
    agent_ids      : lista de agentes
    obs_shapes     : dict[agent_id → int]
    state_dim      : dimensión del estado global para el crítico
    gamma          : factor de descuento
    gae_lambda     : factor λ de GAE (0 = TD, 1 = Monte Carlo)
    """

    def __init__(
        self,
        rollout_length: int,
        agent_ids:      list[str],
        obs_shapes:     dict[str, int],
        state_dim:      int,
        gamma:          float = 0.99,
        gae_lambda:     float = 0.95,
    ):
        self.rollout_length = rollout_length
        self.agents         = agent_ids
        self.state_dim      = state_dim
        self.gamma          = gamma
        self.gae_lambda     = gae_lambda

        self._reset_storage()

    def _reset_storage(self):
        self._obs      = {a: [] for a in self.agents}
        self._actions  = {a: [] for a in self.agents}
        self._log_probs= {a: [] for a in self.agents}
        self._rewards  = {a: [] for a in self.agents}
        self._values   = []      # valor del crítico (compartido)
        self._dones    = []
        self._states   = []
        self._ptr      = 0

    def store(
        self,
        obs:       dict[str, np.ndarray],
        actions:   dict[str, int],
        log_probs: dict[str, float],
        value:     float,
        rewards:   dict[str, float],
        done:      bool,
        state:     np.ndarray,
    ):
        """Almacena un paso del rollout."""
        for a in self.agents:
            self._obs      [a].append(obs      [a].copy())
            self._actions  [a].append(actions  [a])
            self._log_probs[a].append(log_probs[a])
            self._rewards  [a].append(rewards  [a])
        self._values .append(value)
        self._dones  .append(float(done))
        self._states .append(state.copy())
        self._ptr += 1

    def is_full(self) -> bool:
        return self._ptr >= self.rollout_length

    def get(self, last_value: float, device: str = "cpu") -> dict:
        """
        Calcula ventajas (GAE) y devuelve el batch completo.

        Parámetros
        ----------
        last_value : V(s_T) estimado por el crítico en el último estado
        """
        assert self.is_full(), "El buffer no está lleno todavía."

        T      = self.rollout_length
        values = np.array(self._values + [last_value], dtype=np.float32)
        dones  = np.array(self._dones, dtype=np.float32)

        # ── Calcular ventajas con GAE ────────────────────────────────────────
        # GAE(t) = δ_t + (γλ) δ_{t+1} + (γλ)² δ_{t+2} + ...
        # δ_t = r_t + γ V(s_{t+1}) - V(s_t)

        # Usamos la recompensa media de todos los agentes para el crítico compartido
        rewards_mean = np.array([
            np.mean([self._rewards[a][t] for a in self.agents])
            for t in range(T)
        ], dtype=np.float32)

        advantages = np.zeros(T, dtype=np.float32)
        gae        = 0.0
        for t in reversed(range(T)):
            delta = (rewards_mean[t]
                     + self.gamma * values[t+1] * (1 - dones[t])
                     - values[t])
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:T]   # V(s) + ventaja = retorno estimado

        # Normalizar ventajas (reduce varianza, estabiliza PPO)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        def to_t(arr, dtype=torch.float32):
            return torch.tensor(np.array(arr), dtype=dtype).to(device)

        batch = {
            "obs":        {a: to_t(self._obs[a])       for a in self.agents},
            "actions":    {a: to_t(self._actions[a],   dtype=torch.long)
                           for a in self.agents},
            "log_probs":  {a: to_t(self._log_probs[a]) for a in self.agents},
            "advantages": to_t(advantages),
            "returns":    to_t(returns),
            "states":     to_t(self._states),
        }

        self._reset_storage()
        return batch
