"""
networks_nn/mixing_net.py
=========================
Red de mezcla de QMIX.

¿Qué es la mixing network?
  QMIX mezcla los Q-values individuales de cada agente en un Q-total global,
  garantizando que la política conjunta sea consistente con las individuales
  (condición IGM: Individual-Global-Max).

¿Cómo garantiza la monotonicidad?
  Los pesos de la mezcla se generan con un hypernetwork y pasan por abs(),
  lo que los hace siempre positivos. Así:
    ∂Q_tot / ∂Q_i ≥ 0 para todo i
  Es decir, mejorar el Q de un agente nunca empeora el Q_tot.

Arquitectura:
  - Hypernetwork W1: estado → pesos de la primera capa de mezcla
  - Hypernetwork W2: estado → pesos de la segunda capa de mezcla
  - Bias también condicionado al estado

Referencias:
  Rashid et al., "QMIX: Monotonic Value Function Factorisation for
  Deep Multi-Agent Reinforcement Learning", ICML 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QMixer(nn.Module):
    """
    Mixing network de QMIX.

    Parámetros
    ----------
    n_agents       : número de agentes (semáforos)
    state_dim      : dimensión del estado global
    mixing_embed   : dimensión interna de la mixing network
    hypernet_embed : dimensión del hypernetwork que genera los pesos
    """

    def __init__(
        self,
        n_agents:       int,
        state_dim:      int,
        mixing_embed:   int = 32,
        hypernet_embed: int = 64,
    ):
        super().__init__()

        self.n_agents     = n_agents
        self.state_dim    = state_dim
        self.mixing_embed = mixing_embed

        # ── Hypernetwork para W1: estado → pesos de la capa 1 ────────────────
        # W1 tiene shape [n_agents, mixing_embed], aplanada a n_agents*mixing_embed
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, n_agents * mixing_embed),
        )

        # ── Hypernetwork para W2: estado → pesos de la capa 2 ────────────────
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, mixing_embed),
        )

        # ── Sesgos condicionados al estado (no necesitan ser positivos) ───────
        self.hyper_b1 = nn.Linear(state_dim, mixing_embed)

        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, mixing_embed),
            nn.ReLU(),
            nn.Linear(mixing_embed, 1),
        )

    def forward(
        self,
        q_values: torch.Tensor,   # [batch, n_agents]  Q individual de cada agente
        state:    torch.Tensor,   # [batch, state_dim] estado global
    ) -> torch.Tensor:
        """
        Mezcla los Q-values individuales en un Q_total.

        Returns
        -------
        q_tot : [batch, 1]  Q-value total de la política conjunta
        """
        batch = q_values.size(0)

        # Reshape q_values para la operación matricial: [batch, 1, n_agents]
        q = q_values.unsqueeze(1)

        # ── Capa 1 ──────────────────────────────────────────────────────────
        # w1: [batch, n_agents, mixing_embed]  — abs() garantiza monotonicidad
        w1 = torch.abs(self.hyper_w1(state)).view(batch, self.n_agents, self.mixing_embed)
        b1 = self.hyper_b1(state).view(batch, 1, self.mixing_embed)

        # q @ w1 → [batch, 1, mixing_embed]
        hidden = F.elu(torch.bmm(q, w1) + b1)

        # ── Capa 2 ──────────────────────────────────────────────────────────
        # w2: [batch, mixing_embed, 1]  — abs() garantiza monotonicidad
        w2 = torch.abs(self.hyper_w2(state)).view(batch, self.mixing_embed, 1)
        b2 = self.hyper_b2(state).view(batch, 1, 1)

        # hidden @ w2 → [batch, 1, 1]
        q_tot = torch.bmm(hidden, w2) + b2
        return q_tot.squeeze(-1)    # [batch, 1]
