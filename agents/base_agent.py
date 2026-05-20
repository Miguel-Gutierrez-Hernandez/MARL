"""
agents/base_agent.py
====================
Clase abstracta que define la interfaz común para todos los algoritmos MARL.

Todos los agentes implementan:
  - act(obs, explore)     → dict de acciones
  - store_transition(...) → guarda una experiencia en el buffer
  - update()             → un paso de entrenamiento; devuelve las pérdidas
  - save(path)           → serializa el agente a disco
  - load(path)           → carga el agente desde disco
  - eval() / train()     → modo evaluación / entrenamiento
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


class BaseAgent(abc.ABC):
    """
    Interfaz común para todos los algoritmos MARL del proyecto.

    Parámetros
    ----------
    env    : el entorno MultiAgentSumoEnv
    config : diccionario con hiperparámetros (cargado del .yaml)
    device : "cpu", "cuda" o "mps"
    """

    def __init__(self, env, config: dict, device: str = "cpu"):
        self.env     = env
        self.config  = config
        self.device  = device
        self._training = True  # True = modo entrenamiento, False = evaluación

        # Información del entorno (disponible tras reset())
        self.agents    : list[str]      = env.agents
        self.obs_shapes: dict[str, int] = {
            a: env.observation_spaces[a].shape[0] for a in env.agents
        }
        self.n_actions : dict[str, int] = {
            a: env.action_spaces[a].n for a in env.agents
        }
        self.state_dim : int = env.global_state_size

    # ── Interfaz obligatoria ──────────────────────────────────────────────────

    @abc.abstractmethod
    def act(
        self,
        obs:     dict[str, np.ndarray],
        explore: bool = True,
    ) -> dict[str, int]:
        """
        Selecciona una acción para cada agente.

        Parámetros
        ----------
        obs     : observaciones actuales de cada agente
        explore : si True, aplica exploración (ε-greedy, ruido…)
                  si False, actúa de forma greedy (para evaluación)

        Returns
        -------
        actions : dict[agent_id → acción (int)]
        """
        ...

    @abc.abstractmethod
    def store_transition(
        self,
        obs:        dict[str, np.ndarray],
        actions:    dict[str, int],
        rewards:    dict[str, float],
        next_obs:   dict[str, np.ndarray],
        done:       bool,
        state:      Optional[np.ndarray] = None,
        next_state: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """Almacena una transición en el buffer del agente."""
        ...

    @abc.abstractmethod
    def update(self) -> Optional[dict[str, float]]:
        """
        Realiza un paso de actualización de los parámetros.

        Returns
        -------
        losses : dict con las pérdidas registradas (p.ej. {"loss_q": 0.05})
                 o None si no hay suficientes datos para entrenar.
        """
        ...

    # ── Guardar y cargar ──────────────────────────────────────────────────────

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """Serializa el estado del agente a un archivo .pt"""
        ...

    @abc.abstractmethod
    def load(self, path: str) -> None:
        """Carga el estado del agente desde un archivo .pt"""
        ...

    # ── Modo entrenamiento / evaluación ───────────────────────────────────────

    def train(self) -> "BaseAgent":
        """Activa el modo entrenamiento (con exploración y gradientes)."""
        self._training = True
        for module in self._get_all_modules():
            module.train()
        return self

    def eval(self) -> "BaseAgent":
        """Activa el modo evaluación (sin exploración ni gradientes)."""
        self._training = False
        for module in self._get_all_modules():
            module.eval()
        return self

    def _get_all_modules(self) -> list[torch.nn.Module]:
        """Devuelve todos los módulos PyTorch del agente para train()/eval()."""
        modules = []
        for attr in vars(self).values():
            if isinstance(attr, torch.nn.Module):
                modules.append(attr)
            elif isinstance(attr, dict):
                for v in attr.values():
                    if isinstance(v, torch.nn.Module):
                        modules.append(v)
        return modules

    # ── Utilidades comunes ────────────────────────────────────────────────────

    def _to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=torch.float32).to(self.device)

    def _soft_update(
        self,
        online: torch.nn.Module,
        target: torch.nn.Module,
        tau:    float,
    ) -> None:
        """
        Actualización suave (Polyak averaging) de la red objetivo:
        θ_target ← τ·θ_online + (1-τ)·θ_target
        Usado en MADDPG y opcionalmente en QMIX.
        """
        for p_online, p_target in zip(online.parameters(), target.parameters()):
            p_target.data.copy_(tau * p_online.data + (1 - tau) * p_target.data)

    def _hard_update(
        self,
        online: torch.nn.Module,
        target: torch.nn.Module,
    ) -> None:
        """Copia directa de pesos: θ_target ← θ_online. Usado en DQN/IQL."""
        target.load_state_dict(online.state_dict())
