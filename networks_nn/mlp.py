"""
networks_nn/mlp.py
==================
Red neuronal MLP (Multi-Layer Perceptron) compartida por todos los algoritmos.

¿Por qué una sola clase?
  Todos los agentes MARL usan MLPs como bloques básicos.
  Tenerla aquí evita duplicar código en cada algoritmo.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    MLP de profundidad configurable con activaciones ReLU.

    Entrada  : [batch_size, input_dim]
    Salida   : [batch_size, output_dim]

    Parámetros
    ----------
    input_dim  : dimensión del vector de entrada (tamaño de la observación)
    output_dim : dimensión de la salida (nº acciones, 1 para valor, etc.)
    hidden_dim : neuronas en cada capa oculta
    n_layers   : número de capas ocultas (mínimo 1)
    """

    def __init__(
        self,
        input_dim:  int,
        output_dim: int,
        hidden_dim: int = 64,
        n_layers:   int = 2,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim

        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

        # Inicialización ortogonal: más estable que la por defecto en RL
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class QNetwork(MLP):
    """
    Red Q para DQN / IQL / QMIX.
    Salida: Q-value para cada acción discreta.
    """
    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__(obs_dim, n_actions, hidden_dim, n_layers=2)


class ActorNetwork(nn.Module):
    """
    Actor para MAPPO / MADDPG.
    Salida: distribución de probabilidad sobre acciones (softmax).
    Incluye un parámetro de ID de agente concatenado a la entrada,
    lo que permite compartir pesos entre agentes (parameter sharing).
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 64):
        super().__init__()
        self.net = MLP(obs_dim, hidden_dim, hidden_dim, n_layers=1)
        self.head = nn.Linear(hidden_dim, n_actions)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.constant_(self.head.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        """Devuelve una distribución Categorical (para muestrear acciones)."""
        features = F.relu(self.net(obs))
        logits = self.head(features)
        return torch.distributions.Categorical(logits=logits)

    def get_action(self, obs: torch.Tensor):
        """Muestrea una acción y devuelve (acción, log_prob, entropía)."""
        dist = self.forward(obs)
        action   = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()
        return action, log_prob, entropy

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """Para PPO: evalúa log_prob y entropía de acciones ya tomadas."""
        dist     = self.forward(obs)
        log_prob = dist.log_prob(actions)
        entropy  = dist.entropy()
        return log_prob, entropy


class CriticNetwork(MLP):
    """
    Crítico centralizado para MAPPO / MADDPG.
    Entrada: estado global (concatenación de todas las observaciones).
    Salida : valor escalar V(s) o Q(s,a).
    """
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__(state_dim, 1, hidden_dim, n_layers=2)
        # Inicialización más pequeña para la capa de salida del crítico
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)
