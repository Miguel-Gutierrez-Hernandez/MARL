"""
env/sumo_env.py
===============
Entorno multi-agente que conecta SUMO con Gymnasium/PettingZoo.

Arquitectura CTDE (Centralized Training, Decentralized Execution):
  - Durante el entrenamiento: los algoritmos pueden ver el estado global.
  - Durante la ejecución:     cada agente solo usa su observación local.

Este entorno implementa la interfaz ParallelEnv de PettingZoo, en la que
TODOS los agentes actúan simultáneamente en cada paso de tiempo.

Flujo de un episodio:
  1. reset()     → arranca SUMO, genera demanda, devuelve observaciones iniciales
  2. step(actions) (×N) → aplica acciones, avanza simulación delta_time segundos,
                           devuelve obs, rewards, terminated, truncated, info
  3. close()     → cierra SUMO limpiamente

¿Cómo se comunica Python con SUMO?
  A través de TraCI (Traffic Control Interface): un protocolo TCP que permite
  leer y escribir el estado de la simulación en tiempo real.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym

# ── Importar TraCI (incluido en eclipse-sumo) ────────────────────────────────
try:
    import traci
    import sumo
    SUMO_HOME = Path(sumo.__file__).parent
    SUMO_BIN  = SUMO_HOME / "bin" / ("sumo-gui" if sys.platform == "win32"
                                     else "sumo")
    SUMO_BIN_NOGUI = SUMO_HOME / "bin" / ("sumo.exe" if sys.platform == "win32"
                                           else "sumo")
    # Fallback: sistema
    if not SUMO_BIN.exists():
        SUMO_BIN       = Path("sumo-gui")
        SUMO_BIN_NOGUI = Path("sumo")
except ImportError:
    raise ImportError(
        "SUMO no está instalado. Ejecuta: pip install eclipse-sumo"
    )

from env.traffic_signal import TrafficSignal
from env.demand import generate_demand


class MultiAgentSumoEnv:
    """
    Entorno multi-agente SUMO compatible con la interfaz de PettingZoo ParallelEnv.

    Cada semáforo de la red es un agente independiente. Todos actúan al mismo
    tiempo (entorno paralelo), lo que es el modelo estándar en MARL de tráfico.

    Parámetros
    ----------
    network     : topología de la red ("2x2", "3x3", "4x4")
    demand      : nivel de tráfico ("low", "moderate", "saturated")
    num_seconds : duración del episodio en segundos simulados
    delta_time  : segundos entre decisiones de los agentes
    yellow_time : duración de la fase amarilla de transición
    min_green   : tiempo mínimo que debe durar una fase verde
    max_green   : tiempo máximo que puede durar una fase verde
    use_gui     : abrir interfaz gráfica de SUMO (solo para depuración)
    sumo_seed   : semilla de SUMO para reproducibilidad
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        network:     str  = "2x2",
        demand:      str  = "moderate",
        num_seconds: int  = 3600,
        delta_time:  int  = 5,
        yellow_time: int  = 2,
        min_green:   int  = 5,
        max_green:   int  = 60,
        use_gui:     bool = False,
        sumo_seed:   int  = 42,
    ):
        self.network     = network
        self.demand      = demand
        self.num_seconds = num_seconds
        self.delta_time  = delta_time
        self.yellow_time = yellow_time
        self.min_green   = min_green
        self.max_green   = max_green
        self.use_gui     = use_gui
        self.sumo_seed   = sumo_seed

        # Rutas de archivos SUMO
        nets_dir    = Path(__file__).parent.parent / "networks" / network
        self._net   = str(nets_dir / "grid.net.xml")
        self._cfg   = str(nets_dir / "grid.sumocfg")
        self._rou   = str(nets_dir / "grid.rou.xml")

        if not Path(self._net).exists():
            raise FileNotFoundError(
                f"Red no encontrada: {self._net}\n"
                f"Ejecuta primero: python networks/generate_networks.py"
            )

        # Estado interno
        self._sumo_running    : bool = False
        self._step_count      : int  = 0
        self._arrived_total   : int  = 0   # acumulado del episodio
        self.traffic_signals  : dict[str, TrafficSignal] = {}

        # Estos se rellenan en reset() una vez SUMO está corriendo
        self.agents           : list[str] = []
        self.possible_agents  : list[str] = []

        # Espacios de acción y observación (se inicializan en reset)
        self.observation_spaces : dict[str, gym.Space] = {}
        self.action_spaces      : dict[str, gym.Space] = {}
        self._max_obs_size      : int = 0

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed:    int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, np.ndarray], dict]:
        """
        Reinicia la simulación y devuelve las observaciones iniciales.

        Returns
        -------
        observations : dict[agent_id → np.ndarray]
        info         : dict con información adicional (vacío al reset)
        """
        # Cerrar SUMO si ya estaba corriendo
        if self._sumo_running:
            traci.close()
            self._sumo_running = False

        self._arrived_total = 0   # reiniciar contador de vehículos llegados

        # Generar demanda de tráfico para este episodio
        ep_seed = seed if seed is not None else self.sumo_seed
        generate_demand(
            net_xml_path    = self._net,
            output_rou_path = self._rou,
            demand_level    = self.demand,
            num_seconds     = self.num_seconds,
            seed            = ep_seed,
            verbose         = False,
        )

        # Arrancar SUMO
        self._start_sumo(seed=ep_seed)
        self._step_count = 0

        # Descubrir y construir los agentes (semáforos)
        self._build_traffic_signals()

        # Definir espacios de Gymnasium
        self._build_spaces()

        # Observaciones iniciales (un paso de simulación para tener datos)
        traci.simulationStep()
        self._arrived_total += traci.simulation.getArrivedNumber()
        observations = {
            agent_id: self._pad_observation(ts.observation)
            for agent_id, ts in self.traffic_signals.items()
        }

        return observations, {}

    def _start_sumo(self, seed: int) -> None:
        """Lanza el proceso SUMO y abre la conexión TraCI."""
        binary = str(SUMO_BIN if self.use_gui else SUMO_BIN_NOGUI)

        cmd = [
            binary,
            "-c", self._cfg,
            "--route-files", self._rou,
            "--seed", str(seed),
            "--time-to-teleport", "-1",    # Desactivar teleportación (más realista)
            "--waiting-time-memory", str(self.num_seconds),
            "--no-step-log", "true",
            "--no-warnings", "true",
        ]

        # Puerto aleatorio para evitar conflictos si hay múltiples instancias
        port = traci.getFreeSocketPort()
        traci.start(cmd, port=port)
        self._sumo_running = True

    def _build_traffic_signals(self) -> None:
        """Crea un objeto TrafficSignal por cada semáforo de la red."""
        self.traffic_signals = {}
        ts_ids = traci.trafficlight.getIDList()

        for ts_id in ts_ids:
            self.traffic_signals[ts_id] = TrafficSignal(
                ts_id             = ts_id,
                delta_time        = self.delta_time,
                yellow_time       = self.yellow_time,
                min_green         = self.min_green,
                max_green         = self.max_green,
                traci_connection  = traci,
            )

        self.agents          = list(self.traffic_signals.keys())
        self.possible_agents = self.agents.copy()

    def _build_spaces(self) -> None:
        """Define los espacios de observación y acción para Gymnasium."""
        self.observation_spaces = {}
        self.action_spaces      = {}

        self._max_obs_size = max(
            ts.observation_space_size
            for ts in self.traffic_signals.values()
        )

        for agent_id, ts in self.traffic_signals.items():
            n_actions = ts.action_space_size

            self.observation_spaces[agent_id] = gym.spaces.Box(
                low   = 0.0,
                high  = 1.0,
                shape = (self._max_obs_size,),
                dtype = np.float32,
            )
            self.action_spaces[agent_id] = gym.spaces.Discrete(n_actions)

    def _pad_observation(self, obs: np.ndarray) -> np.ndarray:
        """Ajusta la observación al tamaño máximo padding con ceros."""
        if obs.shape[0] == self._max_obs_size:
            return obs
        padded = np.zeros((self._max_obs_size,), dtype=np.float32)
        padded[: obs.shape[0]] = obs
        return padded

    # ── Step ──────────────────────────────────────────────────────────────────

    def step(
        self,
        actions: dict[str, int],
    ) -> tuple[
        dict[str, np.ndarray],   # observations
        dict[str, float],        # rewards
        dict[str, bool],         # terminated
        dict[str, bool],         # truncated
        dict[str, Any],          # info
    ]:
        """
        Aplica las acciones de todos los agentes y avanza la simulación.

        Parámetros
        ----------
        actions : dict con la acción (índice de fase) de cada agente

        Returns
        -------
        observations : nuevas observaciones de cada agente
        rewards      : recompensa de cada agente
        terminated   : True si el episodio terminó por condición de fin
        truncated    : True si el episodio terminó por límite de tiempo
        info         : métricas adicionales para logging
        """
        assert self._sumo_running, "Llama a reset() antes de step()"

        # 1. Aplicar acciones: indicar qué fase quiere cada semáforo
        for agent_id, action in actions.items():
            self.traffic_signals[agent_id].apply_action(action)

        # 2. Avanzar la simulación delta_time segundos
        for _ in range(self.delta_time):
            traci.simulationStep()
            self._step_count += 1

            # Completar transiciones amarillas cuando toca
            for ts in self.traffic_signals.values():
                if ts._in_yellow:
                    ts._time_in_phase += 1
                    if ts._time_in_phase >= self.yellow_time:
                        ts.complete_yellow()

        # TraCI devuelve el número de vehículos llegados en este paso.
        # Acumulamos para obtener el total del episodio.
        self._arrived_total += traci.simulation.getArrivedNumber()

        # 3. Calcular recompensas (después de los delta_time segundos)
        rewards = {
            agent_id: ts.reward
            for agent_id, ts in self.traffic_signals.items()
        }

        # 4. Obtener nuevas observaciones
        observations = {
            agent_id: self._pad_observation(ts.observation)
            for agent_id, ts in self.traffic_signals.items()
        }

        # 5. Condiciones de fin
        sim_time    = traci.simulation.getTime()
        done        = sim_time >= self.num_seconds

        terminated  = {agent_id: done for agent_id in self.agents}
        truncated   = {agent_id: False for agent_id in self.agents}

        # 6. Info: métricas de esta intersección
        info = {
            agent_id: ts.metrics()
            for agent_id, ts in self.traffic_signals.items()
        }
        # Añadir métricas globales de red
        info["__global__"] = self._global_metrics()

        return observations, rewards, terminated, truncated, info

    # ── Métricas globales ─────────────────────────────────────────────────────

    def _global_metrics(self) -> dict:
        """Métricas a nivel de toda la red."""
        # Espera instantánea total (suma de todos los vehículos en todos los carriles)
        total_waiting = sum(
            traci.lane.getWaitingTime(lane)
            for ts in self.traffic_signals.values()
            for lane in ts.lanes
        )
        # Cola: vehículos parados ahora mismo
        total_halting = sum(
            traci.lane.getLastStepHaltingNumber(lane)
            for ts in self.traffic_signals.values()
            for lane in ts.lanes
        )
        # ── Normalización por vehículo ────────────────────────────────────────
        # total_waiting es "veh·s" (suma de esperas individuales).
        # Dividir entre nº de vehículos activos → segundos de espera promedio.
        n_vehicles = max(traci.vehicle.getIDCount(), 1)

        sim_time   = traci.simulation.getTime()
        throughput = self._arrived_total / max(sim_time / 3600.0, 1e-6)

        return {
            "sim_time":                sim_time,
            "total_waiting_time":      total_waiting,
            "mean_waiting_per_vehicle": total_waiting / n_vehicles,  # ← métrica útil
            "total_queue":             total_halting,
            "mean_queue_per_agent":    total_halting / max(len(self.traffic_signals), 1),
            "arrived_vehicles":        self._arrived_total,
            "n_vehicles_active":       n_vehicles,
            "throughput":              throughput,
        }

    # ── Estado global (para algoritmos CTDE como QMIX, MAPPO, MADDPG) ────────

    def get_global_state(self) -> np.ndarray:
        """
        Concatena las observaciones de todos los agentes en un vector global.
        Usado por el crítico centralizado de MAPPO/MADDPG y por QMIX.
        """
        return np.concatenate([
            ts.observation for ts in self.traffic_signals.values()
        ], dtype=np.float32)

    @property
    def global_state_size(self) -> int:
        return sum(ts.observation_space_size for ts in self.traffic_signals.values())

    # ── Utilidades ────────────────────────────────────────────────────────────

    def observation_space(self, agent_id: str) -> gym.Space:
        return self.observation_spaces[agent_id]

    def action_space(self, agent_id: str) -> gym.Space:
        return self.action_spaces[agent_id]

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    def close(self) -> None:
        """Cierra la conexión TraCI y el proceso SUMO."""
        if self._sumo_running:
            traci.close()
            self._sumo_running = False

    def __del__(self):
        self.close()

    def __repr__(self) -> str:
        return (
            f"MultiAgentSumoEnv("
            f"network={self.network}, "
            f"demand={self.demand}, "
            f"agents={self.num_agents if self.agents else '?'}, "
            f"delta_time={self.delta_time}s)"
        )
