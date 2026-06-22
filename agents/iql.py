"""
agents/iql.py — Independent Q-Learning (IQL)
=============================================
Baseline del TFM. Cada semáforo aprende con su propio DQN, sin ninguna
coordinación explícita con los demás agentes.

Algoritmo:
  1. Cada agente i tiene su red Q_i(o_i, a_i) y su red objetivo Q̂_i.
  2. Exploración ε-greedy: con prob ε elige acción aleatoria, si no, greedy.
  3. Almacena (o_i, a_i, r_i, o_i', done) en su replay buffer (compartido).
  4. Mini-batch DQN update independiente por agente:
       L_i = E[(r_i + γ max_{a'} Q̂_i(o_i', a') − Q_i(o_i, a_i))²]
  5. Cada target_update_every pasos: Q̂_i ← Q_i (hard update).

¿Por qué es el baseline?
  No hay coordinación → sirve para medir cuánto aportan los otros algoritmos.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

from agents.base_agent import BaseAgent
from networks_nn.mlp import QNetwork
from training.replay_buffer import MultiAgentReplayBuffer


class IQLAgent(BaseAgent):

    def __init__(self, env, config: dict, device: str = "cpu"):
        super().__init__(env, config, device)

        cfg = config.get("iql", config.get("common", {}))

        self.gamma              = cfg.get("gamma",          0.99)
        self.lr                 = cfg.get("lr",             5e-4)
        self.batch_size         = config.get("common", cfg).get("batch_size", 32)
        self.buffer_size        = config.get("common", cfg).get("buffer_size", 10_000)
        self.hidden_dim         = cfg.get("hidden_dim",     64)
        self.epsilon            = cfg.get("epsilon_start",  1.0)
        self.epsilon_end        = cfg.get("epsilon_end",    0.05)
        self.epsilon_decay      = cfg.get("epsilon_decay",  50_000)
        self.target_update_every= cfg.get("target_update_every", 200)
        self.grad_clip          = config.get("common", cfg).get("grad_clip", 10.0)

        self._step = 0   # contador global de actualizaciones

        # ── Redes Q y objetivos (parameter sharing: todos usan la misma red) ─
        # Una sola red Q compartida recibe la observación de cualquier agente.
        # Alternativa: una red por agente (más parámetros, puede ser mejor).
        obs_dim   = list(self.obs_shapes.values())[0]
        n_actions = list(self.n_actions.values())[0]

        self.q_net      = QNetwork(obs_dim, n_actions, self.hidden_dim).to(device)
        self.target_net = QNetwork(obs_dim, n_actions, self.hidden_dim).to(device)
        self._hard_update(self.q_net, self.target_net)
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)

        # ── Buffer compartido entre todos los agentes ─────────────────────────
        self.buffer = MultiAgentReplayBuffer(
            capacity   = self.buffer_size,
            agent_ids  = self.agents,
            obs_shapes = self.obs_shapes,
            n_actions  = self.n_actions,
        )

    # ── Acción ───────────────────────────────────────────────────────────────

    def act(self, obs: dict, explore: bool = True) -> dict[str, int]:
        actions = {}
        for agent_id, o in obs.items():
            if explore and np.random.random() < self.epsilon:
                actions[agent_id] = np.random.randint(self.n_actions[agent_id])
            else:
                with torch.no_grad():
                    o_t = self._to_tensor(o).unsqueeze(0)       # [1, obs_dim]
                    q   = self.q_net(o_t)                        # [1, n_actions]
                    actions[agent_id] = q.argmax(dim=1).item()

        # Decaimiento lineal de epsilon
        if explore:
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay,
            )
        return actions

    # ── Almacenar transición ─────────────────────────────────────────────────

    def store_transition(self, obs, actions, rewards, next_obs, done,
                         state=None, next_state=None, **kwargs):
        self.buffer.store(obs, actions, rewards, next_obs, done)

    # ── Actualización ────────────────────────────────────────────────────────

    def update(self) -> Optional[dict[str, float]]:
        if not self.buffer.ready(self.batch_size):
            return None

        batch = self.buffer.sample(self.batch_size, self.device)

        total_loss = 0.0

        for agent_id in self.agents:
            obs_a      = batch["obs"]     [agent_id]   # [B, obs_dim]
            next_obs_a = batch["next_obs"][agent_id]   # [B, obs_dim]
            actions_a  = batch["actions"] [agent_id]   # [B]
            rewards_a  = batch["rewards"] [agent_id]   # [B]
            done       = batch["done"]                  # [B]

            # Q(o, a) de la red online
            q_vals   = self.q_net(obs_a)                           # [B, n_actions]
            q_chosen = q_vals.gather(1, actions_a.unsqueeze(1)).squeeze(1)  # [B]

            # Bellman target: r + γ max_a' Q̂(o', a')
            with torch.no_grad():
                q_next  = self.target_net(next_obs_a).max(dim=1).values  # [B]
                target  = rewards_a + self.gamma * (1 - done) * q_next

            loss = F.mse_loss(q_chosen, target)
            total_loss += loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
            self.optimizer.step()

        self._step += 1

        # Hard update de la red objetivo
        if self._step % self.target_update_every == 0:
            self._hard_update(self.q_net, self.target_net)

        return {"loss_q": total_loss / len(self.agents)}

    # ── Guardar / cargar ─────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "q_net":    self.q_net.state_dict(),
            "epsilon":  self.epsilon,
            "step":     self._step,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self._hard_update(self.q_net, self.target_net)
        self.epsilon = ckpt.get("epsilon", self.epsilon_end)
        self._step   = ckpt.get("step", 0)
