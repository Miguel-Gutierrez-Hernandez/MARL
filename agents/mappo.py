"""
agents/mappo.py — MAPPO (Multi-Agent Proximal Policy Optimization)
==================================================================
Algoritmo on-policy basado en PPO. Cada agente tiene su propio actor
(política local) y todos comparten un crítico centralizado.

Arquitectura CTDE:
  - Actor  : o_i → π_i(a_i | o_i)   (solo ve su observación local)
  - Crítico: s   → V(s)              (ve el estado global completo)

Algoritmo:
  1. Recolectar rollout_length pasos con la política actual.
  2. Calcular ventajas GAE: A_t = Σ (γλ)^k δ_{t+k}
  3. Actualizar actor con PPO clipped:
       L_clip = E[min(ρ A, clip(ρ, 1±ε) A)]   donde ρ = π_new / π_old
  4. Actualizar crítico con MSE:
       L_V = E[(V(s) - R_t)²]
  5. Repetir ppo_epochs veces sobre el mismo batch.
  6. Descartar el buffer (on-policy).

Ventaja sobre IQL/QMIX:
  El crítico centralizado usa información de todos los agentes durante
  el entrenamiento, pero la política ejecutada es completamente local.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional

from agents.base_agent import BaseAgent
from networks_nn.mlp import ActorNetwork, CriticNetwork
from training.replay_buffer import RolloutBuffer


class MAPPOAgent(BaseAgent):

    def __init__(self, env, config: dict, device: str = "cpu"):
        super().__init__(env, config, device)

        cfg = config.get("mappo", config.get("common", {}))
        com = config.get("common", cfg)

        self.gamma          = cfg.get("gamma",          0.99)
        self.actor_lr       = cfg.get("actor_lr",       3e-4)
        self.critic_lr      = cfg.get("critic_lr",      1e-3)
        self.gae_lambda     = cfg.get("gae_lambda",     0.95)
        self.clip_eps       = cfg.get("clip_eps",       0.2)
        self.ppo_epochs     = cfg.get("ppo_epochs",     10)
        self.value_loss_coef= cfg.get("value_loss_coef",0.5)
        self.entropy_coef   = cfg.get("entropy_coef",   0.01)
        self.rollout_length = cfg.get("rollout_length", 400)
        self.hidden_dim     = cfg.get("hidden_dim",     64)
        self.grad_clip      = com.get("grad_clip",      10.0)

        obs_dim   = list(self.obs_shapes.values())[0]
        n_actions = list(self.n_actions.values())[0]

        # ── Actor compartido (parameter sharing) ──────────────────────────────
        self.actor = ActorNetwork(obs_dim, n_actions, self.hidden_dim).to(device)

        # ── Crítico centralizado (ve estado global) ───────────────────────────
        self.critic = CriticNetwork(self.state_dim, self.hidden_dim).to(device)

        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=self.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # ── Rollout buffer (on-policy: se vacía tras cada update) ─────────────
        self.buffer = RolloutBuffer(
            rollout_length = self.rollout_length,
            agent_ids      = self.agents,
            obs_shapes     = self.obs_shapes,
            state_dim      = self.state_dim,
            gamma          = self.gamma,
            gae_lambda     = self.gae_lambda,
        )

    # ── Acción estocástica ───────────────────────────────────────────────────

    def act(self, obs: dict, explore: bool = True) -> dict[str, int]:
        actions   = {}
        log_probs = {}

        for agent_id, o in obs.items():
            o_t = self._to_tensor(o).unsqueeze(0)   # [1, obs_dim]
            with torch.no_grad():
                if explore:
                    action, log_prob, _ = self.actor.get_action(o_t)
                else:
                    # Modo greedy: acción más probable
                    dist = self.actor(o_t)
                    action   = dist.probs.argmax(dim=1)
                    log_prob = dist.log_prob(action)

            actions  [agent_id] = action.item()
            log_probs[agent_id] = log_prob.item()

        # Guardar log_probs en el agente para poder pasarlos a store_transition
        self._last_log_probs = log_probs
        return actions

    def store_transition(self, obs, actions, rewards, next_obs, done,
                         state=None, next_state=None, **kwargs):
        """
        MAPPO usa RolloutBuffer. Necesita también el valor V(s) y los log_probs.
        Los log_probs los guardamos en act() para simplificar la interfaz.
        """
        log_probs = getattr(self, "_last_log_probs", {a: 0.0 for a in self.agents})

        with torch.no_grad():
            s_t   = self._to_tensor(state).unsqueeze(0)  # [1, state_dim]
            value = self.critic(s_t).item()

        self.buffer.store(
            obs       = obs,
            actions   = actions,
            log_probs = log_probs,
            value     = value,
            rewards   = rewards,
            done      = done,
            state     = state,
        )

    # ── Actualización PPO ────────────────────────────────────────────────────

    def update(self) -> Optional[dict[str, float]]:
        if not self.buffer.is_full():
            return None

        # Valor del último estado (para el bootstrap de GAE)
        last_obs   = self.env.get_global_state()
        with torch.no_grad():
            last_value = self.critic(
                self._to_tensor(last_obs).unsqueeze(0)
            ).item()

        batch = self.buffer.get(last_value, device=self.device)

        # ── Múltiples épocas PPO sobre el mismo batch ─────────────────────────
        total_actor_loss  = 0.0
        total_critic_loss = 0.0
        total_entropy     = 0.0

        for _ in range(self.ppo_epochs):

            # ── Crítico ───────────────────────────────────────────────────────
            states  = batch["states"]          # [T, state_dim]
            returns = batch["returns"]         # [T]

            values  = self.critic(states).squeeze(1)   # [T]
            v_loss  = F.mse_loss(values, returns)

            self.critic_opt.zero_grad()
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
            self.critic_opt.step()

            # ── Actor (un paso por agente, parameter sharing) ─────────────────
            advantages = batch["advantages"]   # [T]
            actor_loss_agents = []
            entropy_agents    = []

            for agent_id in self.agents:
                obs_a      = batch["obs"]     [agent_id]   # [T, obs_dim]
                actions_a  = batch["actions"] [agent_id]   # [T]
                old_lp_a   = batch["log_probs"][agent_id]  # [T]

                new_lp, entropy = self.actor.evaluate_actions(obs_a, actions_a)

                # Ratio π_new / π_old
                ratio = torch.exp(new_lp - old_lp_a)

                # PPO clipped objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
                actor_loss_i = -torch.min(surr1, surr2).mean()

                actor_loss_agents.append(actor_loss_i)
                entropy_agents.append(entropy.mean())

            actor_loss = torch.stack(actor_loss_agents).mean()
            entropy    = torch.stack(entropy_agents).mean()

            # Loss total del actor: política + coeficiente de entropía
            loss = actor_loss - self.entropy_coef * entropy

            self.actor_opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_opt.step()

            total_actor_loss  += actor_loss.item()
            total_critic_loss += v_loss.item()
            total_entropy     += entropy.item()

        e = self.ppo_epochs
        return {
            "loss_actor":  total_actor_loss  / e,
            "loss_critic": total_critic_loss / e,
            "entropy":     total_entropy     / e,
        }

    def save(self, path: str):
        torch.save({
            "actor":  self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
