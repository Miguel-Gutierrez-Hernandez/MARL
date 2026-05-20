"""
training/trainer.py
====================
Bucle de entrenamiento principal. Conecta el entorno (SUMO) con el agente
(cualquiera de los 5 algoritmos) y el logger (TensorBoard).

Flujo de un paso de entrenamiento:
  ┌──────────────────────────────────────────────────────────┐
  │  obs = env.reset()                                       │
  │  loop:                                                   │
  │    actions  = agent.act(obs, explore=True)               │
  │    next_obs, rewards, done, info = env.step(actions)     │
  │    agent.store_transition(obs, actions, rewards, ...)    │
  │    losses = agent.update()          ← puede ser None     │
  │    logger.log_step(losses)                               │
  │    obs = next_obs                                        │
  │    if done: evaluar, guardar checkpoint                  │
  └──────────────────────────────────────────────────────────┘

El Trainer es agnóstico al algoritmo: funciona igual para IQL, QMIX,
MAPPO, MADDPG y CommNet porque todos implementan la misma interfaz
(BaseAgent).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from training.logger import Logger


class Trainer:
    """
    Bucle de entrenamiento multi-agente genérico.

    Parámetros
    ----------
    env     : MultiAgentSumoEnv
    agent   : cualquier subclase de BaseAgent
    config  : diccionario de configuración completo
    log_dir : carpeta para logs y checkpoints
    seed    : semilla de este run
    """

    def __init__(self, env, agent, config: dict, log_dir, seed: int = 42):
        self.env     = env
        self.agent   = agent
        self.config  = config
        self.log_dir = Path(log_dir)
        self.seed    = seed

        train_cfg = config.get("training", {})
        log_cfg   = config.get("logging",  {})
        algo      = config.get("algorithm", "agent")

        self.total_timesteps  = train_cfg.get("total_timesteps",  500_000)
        self.eval_every       = train_cfg.get("eval_every",        10_000)
        self.eval_episodes    = train_cfg.get("eval_episodes",          5)
        self.save_every       = train_cfg.get("save_every",        50_000)

        self.logger = Logger(
            log_dir       = str(self.log_dir),
            algo          = algo,
            use_tb        = log_cfg.get("tensorboard", True),
            use_wandb     = log_cfg.get("wandb",       False),
            wandb_project = log_cfg.get("wandb_project", "marl-traffic"),
            config        = config,
        )

        # Mejor recompensa vista (para guardar el mejor modelo)
        self._best_mean_waiting = float("inf")

    # ── Entrenamiento ────────────────────────────────────────────────────────

    def train(self) -> dict:
        """
        Ejecuta el entrenamiento completo.

        Returns
        -------
        metrics : dict con las métricas finales del último episodio de eval.
        """
        print(f"\n  Entrenando {self.total_timesteps:,} pasos totales...")
        print(f"  Eval cada {self.eval_every:,} | Checkpoint cada {self.save_every:,}\n")

        global_step    = 0
        episode        = 0
        ep_rewards     = {a: 0.0 for a in self.env.agents}
        ep_steps       = 0
        t_train_start  = time.time()
        final_metrics  = {}

        obs, _ = self.env.reset(seed=self.seed)
        state  = self.env.get_global_state()

        while global_step < self.total_timesteps:

            # ── Acción ───────────────────────────────────────────────────────
            actions = self.agent.act(obs, explore=True)

            # ── Paso de simulación ───────────────────────────────────────────
            next_obs, rewards, terminated, truncated, info = self.env.step(actions)
            next_state = self.env.get_global_state()

            done = all(terminated.values()) or all(truncated.values())

            # ── Almacenar transición ─────────────────────────────────────────
            self.agent.store_transition(
                obs        = obs,
                actions    = actions,
                rewards    = rewards,
                next_obs   = next_obs,
                done       = done,
                state      = state,
                next_state = next_state,
            )

            # ── Actualizar redes ─────────────────────────────────────────────
            losses = self.agent.update()
            if losses:
                self.logger.log_step(losses, step=global_step)

            # ── Acumular estadísticas del episodio ───────────────────────────
            for a in self.env.agents:
                ep_rewards[a] += rewards.get(a, 0.0)
            ep_steps   += 1
            global_step += self.env.delta_time   # cada step = delta_time segundos

            obs   = next_obs
            state = next_state

            # ── Fin de episodio ───────────────────────────────────────────────
            if done:
                episode += 1
                mean_reward = np.mean(list(ep_rewards.values()))
                global_info = info.get("__global__", {})

                ep_metrics = {
                    "mean_reward":              mean_reward,
                    "mean_waiting_per_vehicle": global_info.get("mean_waiting_per_vehicle", 0),
                    "mean_queue_per_agent":     global_info.get("mean_queue_per_agent",    0),
                    "throughput":               global_info.get("throughput",              0),
                    "episode_steps":            ep_steps,
                }
                if hasattr(self.agent, "epsilon"):
                    ep_metrics["epsilon"] = self.agent.epsilon

                self.logger.log_episode(ep_metrics, episode=episode)

                if episode % 5 == 0:
                    elapsed = (time.time() - t_train_start) / 60
                    pct     = 100 * global_step / self.total_timesteps
                    wt      = ep_metrics["mean_waiting_per_vehicle"]
                    print(
                        f"  Ep {episode:4d} | paso {global_step:7,} ({pct:4.1f}%) | "
                        f"r={mean_reward:+.2f} | "
                        f"espera/veh={wt:.1f}s | "
                        f"t={elapsed:.1f}min"
                    )

                # Reset para el siguiente episodio
                obs, _    = self.env.reset(seed=self.seed + episode)
                state     = self.env.get_global_state()
                ep_rewards = {a: 0.0 for a in self.env.agents}
                ep_steps   = 0

            # ── Evaluación periódica ──────────────────────────────────────────
            if global_step % self.eval_every < self.env.delta_time:
                eval_metrics = self._evaluate(global_step)
                final_metrics = eval_metrics

                # Guardar mejor modelo (por tiempo de espera medio)
                wt = eval_metrics.get("mean_waiting_time", float("inf"))
                if wt < self._best_mean_waiting:
                    self._best_mean_waiting = wt
                    self._save_checkpoint("best_model.pt")
                    print(f"  ★ Nuevo mejor modelo: espera={wt:.2f}s "
                          f"(paso {global_step:,})")

            # ── Checkpoint periódico ──────────────────────────────────────────
            if global_step % self.save_every < self.env.delta_time:
                self._save_checkpoint(f"ckpt_{global_step}.pt")

        # ── Fin del entrenamiento ─────────────────────────────────────────────
        self._save_checkpoint("final_model.pt")
        self.logger.close()

        total_time = (time.time() - t_train_start) / 60
        print(f"\n  Entrenamiento completado en {total_time:.1f} min")
        print(f"  Mejor tiempo de espera: {self._best_mean_waiting:.2f}s")

        return final_metrics

    # ── Evaluación ───────────────────────────────────────────────────────────

    def _evaluate(self, global_step: int) -> dict:
        """Ejecuta eval_episodes episodios sin exploración y registra métricas."""
        self.agent.eval()

        waiting_per_veh_list = []
        queue_per_agent_list = []
        throughput_list      = []
        fairness_list        = []

        for ep in range(self.eval_episodes):
            obs, _ = self.env.reset(seed=9999 + ep)
            done   = False

            # Acumuladores por paso (para promediar sobre el episodio)
            step_waiting_per_veh  = []
            step_queue_per_agent  = []
            ts_waiting_accumulator = {a: [] for a in self.env.agents}

            while not done:
                with __import__("torch").no_grad():
                    actions = self.agent.act(obs, explore=False)

                obs, _, terminated, truncated, info = self.env.step(actions)
                done = all(terminated.values()) or all(truncated.values())

                g = info.get("__global__", {})
                step_waiting_per_veh.append(g.get("mean_waiting_per_vehicle", 0))
                step_queue_per_agent.append(g.get("mean_queue_per_agent",     0))

                # Equidad: espera por intersección en este paso
                for a in self.env.agents:
                    ts_info = info.get(a, {})
                    n_veh   = max(ts_info.get("total_vehicles", 1), 1)
                    ts_wt   = ts_info.get("waiting_time", 0) / n_veh
                    ts_waiting_accumulator[a].append(ts_wt)

            # Métricas del episodio
            mean_wt       = float(np.mean(step_waiting_per_veh)) if step_waiting_per_veh else 0.0
            mean_queue    = float(np.mean(step_queue_per_agent))  if step_queue_per_agent  else 0.0
            arrived       = self.env._arrived_total
            sim_time      = self.env.delta_time * len(step_waiting_per_veh)
            throughput    = arrived / max(sim_time / 3600.0, 1e-6)

            # Equidad de Jain entre intersecciones (media del episodio por agente)
            per_ts_means = [
                float(np.mean(ts_waiting_accumulator[a])) if ts_waiting_accumulator[a] else 0.0
                for a in self.env.agents
            ]
            fairness = self._jain_fairness(per_ts_means)

            waiting_per_veh_list.append(mean_wt)
            queue_per_agent_list.append(mean_queue)
            throughput_list     .append(throughput)
            fairness_list       .append(fairness)

        eval_metrics = {
            "mean_waiting_time":  float(np.mean(waiting_per_veh_list)),
            "std_waiting_time":   float(np.std (waiting_per_veh_list)),
            "mean_queue_length":  float(np.mean(queue_per_agent_list)),
            "mean_throughput":    float(np.mean(throughput_list)),
            "fairness_index":     float(np.mean(fairness_list)),
        }

        self.logger.log_eval(eval_metrics, step=global_step)
        self.agent.train()

        print(
            f"  [eval @{global_step:,}] "
            f"espera/veh={eval_metrics['mean_waiting_time']:.1f}±"
            f"{eval_metrics['std_waiting_time']:.1f}s | "
            f"cola/int={eval_metrics['mean_queue_length']:.1f} | "
            f"throughput={eval_metrics['mean_throughput']:.0f} veh/h | "
            f"equidad={eval_metrics['fairness_index']:.3f}"
        )
        return eval_metrics

    # ── Utilidades ────────────────────────────────────────────────────────────

    @staticmethod
    def _jain_fairness(values: list[float]) -> float:
        """
        Índice de equidad de Jain: 1.0 = perfectamente equitativo.
        J = (Σx_i)² / (n · Σx_i²)
        """
        if not values:
            return 1.0
        arr = np.array(values, dtype=np.float64) + 1e-9
        return float((arr.sum() ** 2) / (len(arr) * (arr ** 2).sum()))

    def _save_checkpoint(self, filename: str):
        path = self.log_dir / filename
        self.agent.save(str(path))
