"""
networks_nn/comm_net.py
=======================
Módulos de comunicación entre agentes.

CommNet (Sukhbaatar et al., 2016):
  Cada agente envía su estado oculto y recibe la MEDIA de los estados
  de los demás agentes como "mensaje". Simple pero efectivo.

TarMAC (Das et al., 2019):
  Mejora CommNet con ATENCIÓN: cada agente genera una consulta (query)
  y cada mensaje se pondera según su relevancia para el receptor.
  Permite comunicación selectiva y dirigida.

Flujo de comunicación:
  obs → Encoder → h_i → [Comunicación] → h_i' → Decoder → Q-values / acciones

Referencias:
  - CommNet: "Learning Multiagent Communication with Backpropagation", NeurIPS 2016
  - TarMAC: "TarMAC: Targeted Multi-Agent Communication", ICML 2019
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CommNetModule(nn.Module):
    """
    Un paso de comunicación estilo CommNet.

    Cada agente actualiza su estado oculto sumando la media de los
    estados de los demás. No hay parámetros adicionales en este módulo.

    h_i' = GRU(h_i, mean_{j≠i}(h_j))
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        # GRU cell: integra el mensaje en el estado oculto actual
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(
        self,
        hidden: torch.Tensor,   # [n_agents, batch, hidden_dim]
    ) -> torch.Tensor:
        """
        Un paso de comunicación CommNet.

        Returns
        -------
        new_hidden : [n_agents, batch, hidden_dim]  estados actualizados
        """
        n_agents, batch, h_dim = hidden.shape

        # Media de todos los agentes: [batch, hidden_dim]
        mean_hidden = hidden.mean(dim=0)

        # Actualizar cada agente restando su propia contribución (solo promedio de "otros")
        new_hidden_list = []
        for i in range(n_agents):
            # Mensaje para agente i: media de todos MENOS él mismo
            others_sum  = mean_hidden * n_agents - hidden[i]        # [batch, h_dim]
            message_i   = others_sum / max(n_agents - 1, 1)         # media de j≠i

            # GRU integra el mensaje en el estado oculto
            h_new = self.gru(message_i, hidden[i])                  # [batch, h_dim]
            new_hidden_list.append(h_new)

        return torch.stack(new_hidden_list, dim=0)   # [n_agents, batch, hidden_dim]


class TarMACModule(nn.Module):
    """
    Un paso de comunicación estilo TarMAC (con atención).

    Cada agente i genera:
      - una consulta (query) q_i para indicar "qué tipo de información busca"
      - una clave (key) k_j y un valor (value) v_j para "qué ofrece"

    El mensaje que recibe i es: sum_j( softmax(q_i · k_j / sqrt(d)) * v_j )

    h_i' = GRU(h_i, mensaje_i)
    """

    def __init__(self, hidden_dim: int, attention_dim: int = 16):
        super().__init__()
        self.attention_dim = attention_dim

        # Proyecciones para query, key, value
        self.query  = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.key    = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.value  = nn.Linear(hidden_dim, hidden_dim,   bias=False)

        # GRU para integrar el mensaje
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

        self._scale = attention_dim ** -0.5   # 1/sqrt(d) para escalar el producto

    def forward(
        self,
        hidden: torch.Tensor,   # [n_agents, batch, hidden_dim]
    ) -> torch.Tensor:
        """
        Un paso de comunicación TarMAC.

        Returns
        -------
        new_hidden : [n_agents, batch, hidden_dim]
        """
        n_agents, batch, h_dim = hidden.shape

        # Calcular Q, K, V para todos los agentes a la vez
        # hidden reshaped a [n_agents*batch, h_dim] para las proyecciones
        h_flat = hidden.view(n_agents * batch, h_dim)

        Q = self.query(h_flat).view(n_agents, batch, self.attention_dim)
        K = self.key  (h_flat).view(n_agents, batch, self.attention_dim)
        V = self.value(h_flat).view(n_agents, batch, h_dim)

        new_hidden_list = []
        for i in range(n_agents):
            # q_i: [batch, attention_dim] → [batch, 1, attention_dim]
            q_i = Q[i].unsqueeze(1)

            # k_j para todos j: [n_agents, batch, attention_dim]
            # → [batch, n_agents, attention_dim]
            k_all = K.permute(1, 0, 2)
            v_all = V.permute(1, 0, 2)    # [batch, n_agents, h_dim]

            # Puntuaciones de atención: [batch, 1, n_agents]
            scores = torch.bmm(q_i, k_all.transpose(1, 2)) * self._scale

            # Excluir atención del agente consigo mismo (máscara diagonal)
            mask = torch.zeros(batch, 1, n_agents, device=hidden.device)
            mask[:, :, i] = -1e9
            scores = scores + mask

            weights = F.softmax(scores, dim=-1)   # [batch, 1, n_agents]

            # Mensaje: suma ponderada de valores: [batch, 1, h_dim]
            message_i = torch.bmm(weights, v_all).squeeze(1)  # [batch, h_dim]

            h_new = self.gru(message_i, hidden[i])
            new_hidden_list.append(h_new)

        return torch.stack(new_hidden_list, dim=0)
