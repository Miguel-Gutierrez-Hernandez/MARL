"""
agents/commnet.py — CommNet / TarMAC
=====================================
Los agentes se envían mensajes entre sí antes de decidir su acción.

Arquitectura:
  obs_i → Encoder → h_i → [Comunicación × comm_steps] → h_i' → Q-head → a_i

Flujo de información:
  1. Encoder: convierte la observación local en un vector oculto h_i.
  2. Comunicación: durante comm_steps rondas, cada agente actualiza su
     h_i integrando mensajes de los demás (media en CommNet, atención en TarMAC).
  3. Q-head: h_i' → Q(o_i, a) para seleccionar la acción.

Ventaja sobre IQL:
  Los agentes pueden coordinarse explícitamente antes de actuar, lo que
  permite resolver conflictos en intersecciones adyacentes.

Config:
  use_attention: false → CommNet (media simple de mensajes)
  use_attention: true  → TarMAC (atención multiplicativa)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from agents.base_agent import BaseAgent
from networks_nn.mlp import MLP
from networks_nn.comm_net import CommNetModule, TarMACModule
from training.replay_buffer import MultiAgentReplayBuffer


class CommNetAgent(BaseAgent):

    def __init__(self, env, config: dict, device: str = "cpu"):
        super().__init__(env, config, device)

        cfg = config.get("commnet", config.get("common", {}))
        com = config.get("common", cfg)

        self.gamma               = cfg.get("gamma",          0.99)
        self.lr                  = cfg.get("lr",             5e-4)
        self.batch_size          = com.get("batch_size",     32)
        self.buffer_size         = com.get("buffer_size",    10_000)
        self.hidden_dim          = cfg.get("hidden_dim",     64)
        self.comm_dim            = cfg.get("comm_dim",       32)
        self.comm_steps          = cfg.get("comm_steps",     2)
        self.use_attention       = cfg.get("use_attention",  False)
        self.attention_dim       = cfg.get("attention_dim",  16)
        self.epsilon             = cfg.get("epsilon_start",  1.0)
        self.epsilon_end         = cfg.get("epsilon_end",    0.05)
        self.epsilon_decay       = cfg.get("epsilon_decay",  50_000)
        self.target_update_every = cfg.get("target_update_every", 200)
        self.grad_clip           = com.get("grad_clip",      10.0)

        self._update_step = 0
        n_agents          = len(self.agents)
        obs_dim           = list(self.obs_shapes.values())[0]
        n_actions         = list(self.n_actions.values())[0]

        # ── Encoder: obs_dim → hidden_dim ────────────────────────────────────
        self.encoder = MLP(obs_dim, self.hidden_dim, self.hidden_dim,
                           n_layers=1).to(device)

        # ── Módulos de comunicación (apilados comm_steps veces) ───────────────
        CommModule = TarMACModule if self.use_attention else CommNetModule
        comm_kwargs = {"hidden_dim": self.hidden_dim}
        if self.use_attention:
            comm_kwargs["attention_dim"] = self.attention_dim

        self.comm_modules = nn.ModuleList([
            CommModule(**comm_kwargs) for _ in range(self.comm_steps)
        ]).to(device)

        # ── Q-head: hidden_dim → n_actions ───────────────────────────────────
        self.q_head        = nn.Linear(self.hidden_dim, n_actions).to(device)
        self.target_encoder = MLP(obs_dim, self.hidden_dim, self.hidden_dim,
                                  n_layers=1).to(device)
        self.target_q_head  = nn.Linear(self.hidden_dim, n_actions).to(device)
        self._sync_targets()
        self.target_encoder.eval()
        self.target_q_head.eval()

        # Todos los parámetros entrenables en un solo optimizador
        params = (list(self.encoder.parameters())
                + list(self.comm_modules.parameters())
                + list(self.q_head.parameters()))
        self.optimizer = torch.optim.Adam(params, lr=self.lr)

        # ── Buffer ────────────────────────────────────────────────────────────
        self.buffer = MultiAgentReplayBuffer(
            capacity   = self.buffer_size,
            agent_ids  = self.agents,
            obs_shapes = self.obs_shapes,
            n_actions  = self.n_actions,
        )

        self.n_actions_val = n_actions
        self.n_agents      = n_agents

    # ── Forward de la red de comunicación ────────────────────────────────────

    def _forward(
        self,
        obs_dict: dict[str, torch.Tensor],
        use_target: bool = False,
    ) -> torch.Tensor:
        """
        Pasa las observaciones de todos los agentes por la red completa.

        Returns
        -------
        q_values : [n_agents, batch, n_actions]
        """
        enc    = self.target_encoder if use_target else self.encoder
        q_head = self.target_q_head  if use_target else self.q_head

        # Codificar cada agente: [n_agents, batch, hidden_dim]
        hidden = torch.stack(
            [enc(obs_dict[a]) for a in self.agents], dim=0
        )

        # Rondas de comunicación
        for comm in self.comm_modules:
            hidden = comm(hidden)   # [n_agents, batch, hidden_dim]

        # Q-values para cada agente: [n_agents, batch, n_actions]
        q_values = q_head(hidden)
        return q_values

    # ── Acción ε-greedy con comunicación ─────────────────────────────────────

    def act(self, obs: dict, explore: bool = True) -> dict[str, int]:
        # Convertir a tensores
        obs_t = {a: self._to_tensor(obs[a]).unsqueeze(0) for a in self.agents}
        # obs_t[a]: [1, obs_dim]

        with torch.no_grad():
            q_values = self._forward(obs_t)   # [n_agents, 1, n_actions]

        actions = {}
        for i, agent_id in enumerate(self.agents):
            if explore and np.random.random() < self.epsilon:
                actions[agent_id] = np.random.randint(self.n_actions_val)
            else:
                actions[agent_id] = q_values[i, 0].argmax().item()

        if explore:
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay,
            )
        return actions

    def store_transition(self, obs, actions, rewards, next_obs, done,
                         state=None, next_state=None, **kwargs):
        self.buffer.store(obs, actions, rewards, next_obs, done)

    # ── Actualización ────────────────────────────────────────────────────────

    def update(self) -> Optional[dict[str, float]]:
        if not self.buffer.ready(self.batch_size):
            return None

        batch = self.buffer.sample(self.batch_size, self.device)
        B     = self.batch_size

        # Q-values de la red online para todos los agentes: [n_agents, B, n_actions]
        q_all = self._forward(batch["obs"])

        # Q-values de la red objetivo en el siguiente estado
        with torch.no_grad():
            q_next_all = self._forward(batch["next_obs"], use_target=True)
            # max_a Q̂(o', a): [n_agents, B]
            q_next_max = q_next_all.max(dim=2).values

        done = batch["done"]   # [B]
        total_loss = 0.0

        for i, agent_id in enumerate(self.agents):
            action_a = batch["actions"][agent_id]          # [B]
            reward_a = batch["rewards"][agent_id]          # [B]

            # Q(o_i, a_i): [B]
            q_chosen = q_all[i].gather(1, action_a.unsqueeze(1)).squeeze(1)

            # Bellman target
            target   = reward_a + self.gamma * (1 - done) * q_next_max[i]
            total_loss += F.mse_loss(q_chosen, target)

        loss = total_loss / self.n_agents

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.comm_modules.parameters())
            + list(self.q_head.parameters()),
            self.grad_clip,
        )
        self.optimizer.step()

        self._update_step += 1
        if self._update_step % self.target_update_every == 0:
            self._sync_targets()

        comm_type = "TarMAC" if self.use_attention else "CommNet"
        return {f"loss_q_{comm_type.lower()}": loss.item()}

    # ── Sincronización de redes objetivo ─────────────────────────────────────

    def _sync_targets(self):
        self._hard_update(self.encoder, self.target_encoder)
        self._hard_update(self.q_head,  self.target_q_head)

    def save(self, path: str):
        torch.save({
            "encoder":      self.encoder.state_dict(),
            "comm_modules": self.comm_modules.state_dict(),
            "q_head":       self.q_head.state_dict(),
            "epsilon":      self.epsilon,
            "step":         self._update_step,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.comm_modules.load_state_dict(ckpt["comm_modules"])
        self.q_head.load_state_dict(ckpt["q_head"])
        self._sync_targets()
        self.epsilon      = ckpt.get("epsilon", self.epsilon_end)
        self._update_step = ckpt.get("step", 0)
