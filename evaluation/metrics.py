"""
evaluation/metrics.py
======================
Calcula y agrega las 5 métricas de evaluación del TFM:

  1. Tiempo de espera medio       (waiting time)
  2. Longitud de cola media       (queue length)
  3. Paradas por vehículo        (stops per vehicle)
  4. Throughput                  (vehículos/hora que cruzan la red)
  5. Índice de equidad de Jain   (fairness entre intersecciones)

EpisodeMetrics acumula los datos paso a paso y los resume al final.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class EpisodeMetrics:
    """
    Acumula métricas durante un episodio y las resume al final.

    Uso:
        em = EpisodeMetrics()
        while not done:
            _, _, _, _, info = env.step(actions)
            em.update(info)
        summary = em.summarize()
    """

    # Acumuladores por paso
    _waiting_steps:  list[float] = field(default_factory=list)
    _queue_steps:    list[float] = field(default_factory=list)
    _throughput:     list[float] = field(default_factory=list)
    _arrived_total:  int         = 0
    _steps:          int         = 0

    # Métricas por intersección (para equidad)
    _per_ts_waiting: dict[str, list[float]] = field(default_factory=dict)

    def update(self, info: dict[str, Any]) -> None:
        """
        Registra un paso de simulación.

        info : el dict devuelto por env.step()
                incluye métricas por agente y "__global__"
        """
        global_info = info.get("__global__", {})

        self._waiting_steps.append(global_info.get("total_waiting_time", 0.0))
        self._queue_steps  .append(global_info.get("total_queue",        0.0))
        self._arrived_total = int(global_info.get("arrived_vehicles",    0))
        self._steps        += 1

        sim_time = global_info.get("sim_time", self._steps)
        tput = self._arrived_total / max(sim_time / 3600, 1e-6)
        self._throughput.append(tput)

        # Guardar espera por intersección (para calcular equidad)
        for agent_id, agent_info in info.items():
            if agent_id == "__global__":
                continue
            if isinstance(agent_info, dict):
                wt = agent_info.get("waiting_time", 0.0)
                if agent_id not in self._per_ts_waiting:
                    self._per_ts_waiting[agent_id] = []
                self._per_ts_waiting[agent_id].append(wt)

    def summarize(self) -> dict[str, float]:
        """
        Devuelve el resumen del episodio con las 5 métricas del TFM.
        """
        n = max(self._steps, 1)

        # 1. Tiempo de espera medio por paso
        mean_waiting = float(np.mean(self._waiting_steps)) if self._waiting_steps else 0.0

        # 2. Longitud de cola media
        mean_queue = float(np.mean(self._queue_steps)) if self._queue_steps else 0.0

        # 3. Paradas por vehículo
        #    Aproximación: vehículos parados acumulados / total de llegadas
        total_halting = float(np.sum(self._queue_steps))
        stops_per_veh = total_halting / max(self._arrived_total, 1)

        # 4. Throughput (vehículos/hora al final del episodio)
        throughput = float(self._throughput[-1]) if self._throughput else 0.0

        # 5. Índice de equidad de Jain entre intersecciones
        if self._per_ts_waiting:
            ts_mean_waits = [
                float(np.mean(v)) for v in self._per_ts_waiting.values()
            ]
            fairness = _jain_fairness(ts_mean_waits)
        else:
            fairness = 1.0

        return {
            "mean_waiting_time": mean_waiting,
            "mean_queue_length": mean_queue,
            "mean_stops":        stops_per_veh,
            "throughput":        throughput,
            "fairness_index":    fairness,
            "arrived_vehicles":  self._arrived_total,
            "episode_steps":     self._steps,
        }

    def reset(self) -> None:
        """Reinicia el acumulador para un nuevo episodio."""
        self._waiting_steps  = []
        self._queue_steps    = []
        self._throughput     = []
        self._arrived_total  = 0
        self._steps          = 0
        self._per_ts_waiting = {}


def _jain_fairness(values: list[float]) -> float:
    """
    Índice de equidad de Jain ∈ [1/n, 1].
    1.0 = todos los agentes tienen la misma carga.
    """
    arr = np.array(values, dtype=np.float64) + 1e-9
    return float((arr.sum() ** 2) / (len(arr) * (arr ** 2).sum()))
