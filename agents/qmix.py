"""
agents/qmix.py — QMIX
=====================
QMIX factoriza el Q-total en Q-values individuales mezclados de forma
monotónica. Esto garantiza que la acción óptima conjunta coincida con
las acciones óptimas individuales (condición IGM).

Algoritmo:
  1. Cada agente i tiene su Q_i(o_i, a_i) (igual que IQL).
  2. Un Mixer combina los Q_i con el estado global s:
       Q_tot = Mixer(Q_1,...,Q_n, s)   con ∂Q_tot/∂Q_i ≥ 0
  3. Loss:
       L = E[(r_tot + γ max_a Q̂_tot(s', â) − Q_tot(s, a))²]
     donde r_tot = media de recompensas individuales.
  4. Hard update de las redes objetivo cada target_update_every pasos.

Diferencia clave vs IQL:
  El gradiente fluye desde Q_tot hasta cada Q_i durante el entrenamiento.
  Así los agentes aprenden a coordinarse, aunque en ejecución cada uno
  solo ve su observación local.
"""

from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

from agents.base_agent import BaseAgent
from networks_nn.mlp import QNetwork
from networks_nn.mixing_net import QMixer
from training.replay_buffer import MultiAgentReplayBuffer


class QMIXAgent(BaseAgent):

    def __init__(self, env, config: dict, device: str = "cpu"):
        super().__init__(env, config, device)

        cfg = config.get("qmix", config.get("common", {}))
        com = config.get("common", cfg)

        self.gamma               = cfg.get("gamma",                0.99)
        self.lr                  = cfg.get("lr",                   5e-4)
        self.batch_size          = com.get("batch_size",           32)
        self.buffer_size         = com.get("buffer_size",          10_000)
        self.hidden_dim          = cfg.get("hidden_dim",           64)
        self.mixing_embed_dim    = cfg.get("mixing_embed_dim",     32)
        self.hypernet_embed      = cfg.get("hypernet_embed",       64)
        self.epsilon             = cfg.get("epsilon_start",        1.0)
        self.epsilon_end         = cfg.get("epsilon_end",          0.05)
        self.epsilon_decay       = cfg.get("epsilon_decay",        50_000)
        self.target_update_every = cfg.get("target_update_every",  200)
        self.grad_clip           = com.get("grad_clip",            10.0)

        self._step = 0
        n_agents  = len(self.agents)
        obs_dim   = list(self.obs_shapes.values())[0]
        n_actions = list(self.n_actions.values())[0]

        # ── Red Q individual (parameter sharing) ─────────────────────────────
        self.q_net      = QNetwork(obs_dim, n_actions, self.hidden_dim).to(device)
        self.target_q   = QNetwork(obs_dim, n_actions, self.hidden_dim).to(device)
        self._hard_update(self.q_net, self.target_q)
        self.target_q.eval()

        # ── Mixing network y su objetivo ──────────────────────────────────────
        self.mixer        = QMixer(n_agents, self.state_dim,
                                   self.mixing_embed_dim, self.hypernet_embed).to(device)
        self.target_mixer = copy.deepcopy(self.mixer).to(device)
        self.target_mixer.eval()

        # Un solo optimizador para Q + mixer juntos
        self.optimizer = torch.optim.Adam(
            list(self.q_net.parameters()) + list(self.mixer.parameters()),
            lr=self.lr,
        )

        # ── Buffer (guarda estado global para el mixer) ───────────────────────
        self.buffer = MultiAgentReplayBuffer(
            capacity   = self.buffer_size,
            agent_ids  = self.agents,
            obs_shapes = self.obs_shapes,
            n_actions  = self.n_actions,
            state_dim  = self.state_dim,
        )

    # ── Acción ε-greedy (idéntica a IQL) ────────────────────────────────────

    def act(self, obs: dict, explore: bool = True) -> dict[str, int]:
        actions = {}
        for agent_id, o in obs.items():
            if explore and np.random.random() < self.epsilon:
                actions[agent_id] = np.random.randint(self.n_actions[agent_id])
            else:
                with torch.no_grad():
                    o_t = self._to_tensor(o).unsqueeze(0)
                    actions[agent_id] = self.q_net(o_t).argmax(dim=1).item()

        if explore:
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay,
            )
        return actions

    def store_transition(self, obs, actions, rewards, next_obs, done,
                         state=None, next_state=None, **kwargs):
        self.buffer.store(obs, actions, rewards, next_obs, done, state, next_state)

    # ── Actualización ────────────────────────────────────────────────────────

    def update(self) -> Optional[dict[str, float]]:
        if not self.buffer.ready(self.batch_size):
            return None

        batch = self.buffer.sample(self.batch_size, self.device)
        B = self.batch_size

        # ── Q_i(o_i, a_i) para cada agente → [B, n_agents] ──────────────────
        q_chosen = []
        for agent_id in self.agents:
            obs_a    = batch["obs"][agent_id]             # [B, obs_dim]
            action_a = batch["actions"][agent_id]         # [B]
            q_all    = self.q_net(obs_a)                  # [B, n_actions]
            q_a      = q_all.gather(1, action_a.unsqueeze(1)).squeeze(1)  # [B]
            q_chosen.append(q_a)

        q_chosen = torch.stack(q_chosen, dim=1)           # [B, n_agents]

        # ── Q_tot = Mixer(Q_i, s) ─────────────────────────────────────────────
        state      = batch["state"]                        # [B, state_dim]
        q_tot      = self.mixer(q_chosen, state)           # [B, 1]

        # ── Target: r_tot + γ max_a' Q̂_tot(s', â) ───────────────────────────
        with torch.no_grad():
            # max Q̂_i para cada agente
            q_next_list = []
            for agent_id in self.agents:
                next_obs_a = batch["next_obs"][agent_id]
                q_next_a   = self.target_q(next_obs_a).max(dim=1).values
                q_next_list.append(q_next_a)

            q_next   = torch.stack(q_next_list, dim=1)    # [B, n_agents]
            next_state = batch["next_state"]               # [B, state_dim]
            q_tot_next = self.target_mixer(q_next, next_state)  # [B, 1]

            # Recompensa global = media de recompensas individuales
            r_tot  = torch.stack(
                [batch["rewards"][a] for a in self.agents], dim=1
            ).mean(dim=1, keepdim=True)                    # [B, 1]

            done   = batch["done"].unsqueeze(1)            # [B, 1]
            target = r_tot + self.gamma * (1 - done) * q_tot_next

        loss = F.mse_loss(q_tot, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.q_net.parameters()) + list(self.mixer.parameters()),
            self.grad_clip,
        )
        self.optimizer.step()

        self._step += 1
        if self._step % self.target_update_every == 0:
            self._hard_update(self.q_net,  self.target_q)
            self._hard_update(self.mixer,  self.target_mixer)

        return {"loss_q_tot": loss.item()}

    def save(self, path: str):
        torch.save({
            "q_net":   self.q_net.state_dict(),
            "mixer":   self.mixer.state_dict(),
            "epsilon": self.epsilon,
            "step":    self._step,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.mixer.load_state_dict(ckpt["mixer"])
        self._hard_update(self.q_net,  self.target_q)
        self._hard_update(self.mixer,  self.target_mixer)
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
        self._step   = ckpt.get("step", 0)
