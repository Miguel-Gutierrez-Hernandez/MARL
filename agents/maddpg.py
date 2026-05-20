"""
agents/maddpg.py — MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
===========================================================================
MADDPG extiende DDPG al entorno multi-agente con críticos centralizados.

Arquitectura CTDE:
  - Actor_i  : o_i → π_i(a_i)           (política local de cada agente)
  - Crítico_i: (o_1,...,o_n, a_1,...,a_n) → Q_i   (ve todo, evalúa agente i)

El espacio de acción de los semáforos es DISCRETO, pero DDPG necesita
acciones continuas y diferenciables. Solución: Gumbel-Softmax.

Gumbel-Softmax:
  En lugar de π_i → one-hot, el actor produce un vector de probabilidades
  suavizado por temperatura τ. Al pasar por argmax se discretiza, pero
  el gradiente fluye por la aproximación "straight-through".

Algoritmo:
  1. Actor_i elige acción a_i = Gumbel-Softmax(o_i) + ruido de exploración.
  2. Buffer guarda (o, a, r, o', done) de todos los agentes.
  3. Actualizar crítico_i (TD con redes objetivo):
       L_i = E[(r_i + γ Q̂_i(o', â') − Q_i(o, a))²]
  4. Actualizar actor_i (gradiente del crítico):
       L_actor_i = -E[Q_i(o, π_i(o_i), a_{j≠i})]
  5. Soft update de redes objetivo: θ̂ ← τθ + (1-τ)θ̂
"""

from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

from agents.base_agent import BaseAgent
from networks_nn.mlp import MLP
from training.replay_buffer import MultiAgentReplayBuffer


class MADDPGAgent(BaseAgent):

    def __init__(self, env, config: dict, device: str = "cpu"):
        super().__init__(env, config, device)

        cfg = config.get("maddpg", config.get("common", {}))
        com = config.get("common", cfg)

        self.gamma       = cfg.get("gamma",       0.99)
        self.actor_lr    = cfg.get("actor_lr",    1e-4)
        self.critic_lr   = cfg.get("critic_lr",   1e-3)
        self.tau         = cfg.get("tau",         0.01)
        self.noise_std   = cfg.get("noise_std",   0.1)
        self.batch_size  = cfg.get("batch_size",  64)
        self.buffer_size = cfg.get("buffer_size", 100_000)
        self.warmup_steps= cfg.get("warmup_steps",1000)
        self.hidden_dim  = cfg.get("hidden_dim",  64)
        self.grad_clip   = com.get("grad_clip",   10.0)

        self._step    = 0
        n_agents      = len(self.agents)
        obs_dim       = list(self.obs_shapes.values())[0]
        n_actions     = list(self.n_actions.values())[0]

        # El crítico ve todas las obs + todas las acciones (one-hot)
        critic_input_dim = obs_dim * n_agents + n_actions * n_agents

        # ── Actores y críticos (parameter sharing) ────────────────────────────
        # Actor: o_i → logits sobre acciones (aplicamos Gumbel-Softmax)
        self.actor        = MLP(obs_dim,         n_actions, self.hidden_dim).to(device)
        self.target_actor = copy.deepcopy(self.actor).to(device)

        # Crítico: (all_obs, all_actions) → Q escalar
        self.critic        = MLP(critic_input_dim, 1, self.hidden_dim).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)

        self.target_actor.eval()
        self.target_critic.eval()

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=self.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # ── Buffer ────────────────────────────────────────────────────────────
        self.buffer = MultiAgentReplayBuffer(
            capacity   = self.buffer_size,
            agent_ids  = self.agents,
            obs_shapes = self.obs_shapes,
            n_actions  = self.n_actions,
        )

        self.n_actions_val = n_actions
        self.obs_dim       = obs_dim

    # ── Acción con Gumbel-Softmax + ruido ─────────────────────────────────────

    def act(self, obs: dict, explore: bool = True) -> dict[str, int]:
        actions = {}
        for agent_id, o in obs.items():
            o_t    = self._to_tensor(o).unsqueeze(0)   # [1, obs_dim]
            with torch.no_grad():
                logits = self.actor(o_t)               # [1, n_actions]

            if explore and self._step < self.warmup_steps:
                # Exploración totalmente aleatoria al inicio
                actions[agent_id] = np.random.randint(self.n_actions[agent_id])
            elif explore:
                # Gumbel-Softmax con temperatura para exploración
                gumbel = F.gumbel_softmax(logits, tau=1.0, hard=True)
                # Añadir ruido gaussiano pequeño antes del argmax
                noisy  = gumbel + torch.randn_like(gumbel) * self.noise_std
                actions[agent_id] = noisy.argmax(dim=1).item()
            else:
                actions[agent_id] = logits.argmax(dim=1).item()

        self._step += 1
        return actions

    def store_transition(self, obs, actions, rewards, next_obs, done,
                         state=None, next_state=None, **kwargs):
        self.buffer.store(obs, actions, rewards, next_obs, done)

    # ── Actualización ────────────────────────────────────────────────────────

    def update(self) -> Optional[dict[str, float]]:
        if not self.buffer.ready(self.batch_size):
            return None

        batch = self.buffer.sample(self.batch_size, self.device)
        B = self.batch_size

        # Construir tensores globales: all_obs, all_actions (one-hot)
        all_obs      = torch.cat([batch["obs"][a]      for a in self.agents], dim=1)  # [B, obs*N]
        all_next_obs = torch.cat([batch["next_obs"][a] for a in self.agents], dim=1)

        # One-hot de las acciones actuales
        def to_onehot(action_tensor):
            oh = torch.zeros(B, self.n_actions_val, device=self.device)
            oh.scatter_(1, action_tensor.unsqueeze(1), 1.0)
            return oh

        all_actions_oh = torch.cat(
            [to_onehot(batch["actions"][a]) for a in self.agents], dim=1
        )   # [B, n_actions*N]

        # ── Actualizar crítico ────────────────────────────────────────────────
        with torch.no_grad():
            # Acciones de los actores objetivo en el siguiente estado
            next_actions_oh_list = []
            for a in self.agents:
                next_logits = self.target_actor(batch["next_obs"][a])
                next_oh     = to_onehot(next_logits.argmax(dim=1))
                next_actions_oh_list.append(next_oh)
            all_next_actions_oh = torch.cat(next_actions_oh_list, dim=1)

            critic_next_input = torch.cat([all_next_obs, all_next_actions_oh], dim=1)
            # Para simplificar (parameter sharing): mismo crítico para todos
            q_next = self.target_critic(critic_next_input).squeeze(1)  # [B]

        # Recompensa media de todos los agentes
        r_mean = torch.stack([batch["rewards"][a] for a in self.agents], dim=1).mean(1)
        done   = batch["done"]
        target = r_mean + self.gamma * (1 - done) * q_next              # [B]

        critic_input = torch.cat([all_obs, all_actions_oh], dim=1)
        q_val = self.critic(critic_input).squeeze(1)                    # [B]
        critic_loss = F.mse_loss(q_val, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_opt.step()

        # ── Actualizar actor ──────────────────────────────────────────────────
        # Acciones del actor actual (diferenciables via Gumbel-Softmax)
        curr_actions_oh_list = []
        for a in self.agents:
            logits = self.actor(batch["obs"][a])
            oh     = F.gumbel_softmax(logits, tau=1.0, hard=True)  # diferenciable
            curr_actions_oh_list.append(oh)
        all_curr_actions_oh = torch.cat(curr_actions_oh_list, dim=1)

        actor_input = torch.cat([all_obs, all_curr_actions_oh], dim=1)
        actor_loss  = -self.critic(actor_input).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_opt.step()

        # ── Soft update de redes objetivo ─────────────────────────────────────
        self._soft_update(self.actor,  self.target_actor,  self.tau)
        self._soft_update(self.critic, self.target_critic, self.tau)

        return {
            "loss_critic": critic_loss.item(),
            "loss_actor":  actor_loss.item(),
        }

    def save(self, path: str):
        torch.save({
            "actor":  self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "step":   self._step,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self._hard_update(self.actor,  self.target_actor)
        self._hard_update(self.critic, self.target_critic)
        self._step = ckpt.get("step", 0)
