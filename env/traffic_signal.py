"""
env/traffic_signal.py
======================
Representa un único semáforo como un agente de RL.

Conceptos clave:
  - observation : vector numérico que el agente "ve" de su entorno local
  - action      : qué fase activar a continuación (número entero)
  - reward      : señal de refuerzo basada en la reducción de tiempos de espera

¿Qué es una "fase" de semáforo en SUMO?
  Una cadena de caracteres como "GGrrGGrr" donde:
    G = verde, g = verde cede el paso, r = rojo, y = amarillo
  Cada posición corresponde a una conexión entre carriles.

Transición entre fases:
  Cuando el agente quiere cambiar de fase, primero se activa una fase
  AMARILLA intermedia (por seguridad), y luego la nueva fase verde.
  Este comportamiento está aquí implementado.
"""

from __future__ import annotations

import numpy as np


# Longitud máxima de carril para normalizar densidad (metros)
MAX_LANE_LENGTH = 300.0
# Velocidad mínima para considerar un vehículo "parado" (m/s)
MIN_SPEED_THRESHOLD = 0.1


class TrafficSignal:
    """
    Abstracción de un semáforo como agente MARL.

    Atributos principales
    ---------------------
    id          : identificador del semáforo en SUMO (ej. "cluster_0_1")
    traci       : conexión activa con SUMO (inyectada desde el entorno)
    green_phases : índices de las fases verdes (excluye fases amarillas)
    """

    def __init__(
        self,
        ts_id: str,
        delta_time: int,
        yellow_time: int,
        min_green: int,
        max_green: int,
        traci_connection,
    ):
        self.id          = ts_id
        self.delta_time  = delta_time    # Segundos entre decisiones
        self.yellow_time = yellow_time   # Duración de la fase amarilla
        self.min_green   = min_green     # Verde mínimo por fase
        self.max_green   = max_green     # Verde máximo por fase
        self.traci       = traci_connection

        # Estado interno del semáforo
        self._current_phase   : int  = 0
        self._time_in_phase   : int  = 0    # Segundos en la fase actual
        self._in_yellow       : bool = False
        self._next_phase      : int  = 0    # Fase destino tras el amarillo
        self._last_waiting    : float = 0.0  # Para calcular la recompensa

        # Caché de la lógica de semáforos (se llena en _setup)
        self.green_phases     : list[int] = []
        self.lanes            : list[str] = []
        self.out_lanes = []
        self.num_green_phases : int = 0

        self._setup()

    # ── Inicialización ─────────────────────────────────────────────────────────

    def _setup(self) -> None:
        """Lee de SUMO la lógica del semáforo y los carriles que controla."""
        # Obtener la lógica de semáforo del nodo
        logic = self.traci.trafficlight.getAllProgramLogics(self.id)[0]
        all_phases = logic.phases

        # Identificar fases verdes (las que contienen 'G' o 'g' pero no 'y'/'Y')
        self.green_phases = [
            i for i, p in enumerate(all_phases)
            if ("G" in p.state or "g" in p.state)
            and "y" not in p.state.lower()
        ]
        self.num_green_phases = len(self.green_phases)

        # Carriles controlados por este semáforo
        controlled_links = self.traci.trafficlight.getControlledLinks(self.id)
        seen = set()
        for link_list in controlled_links:
            for link in link_list:
                if link and link[0] not in seen:
                    self.lanes.append(link[0])   # link[0] = carril de entrada
                    seen.add(link[0])

        # Número de carriles por fase (para el espacio de observación)
        self._num_lanes = len(self.lanes)

        # Fase inicial
        self._current_phase = self.green_phases[0] if self.green_phases else 0
        self.traci.trafficlight.setPhase(self.id, self._current_phase)

    # ── Espacio de observación ─────────────────────────────────────────────────

    @property
    def observation_space_size(self) -> int:
        """
        Tamaño del vector de observación:
          - densidad por carril      (num_lanes valores en [0,1])
          - cola normalizada          (num_lanes valores en [0,1])
          - fase actual (one-hot)    (num_green_phases valores en {0,1})
          - tiempo en fase normaliz. (1 valor en [0,1])
        """
        return self._num_lanes * 2 + self.num_green_phases + 1

    @property
    def observation(self) -> np.ndarray:
        densities = self._lane_densities()
        queues = self._lane_queues()
        phase_onehot = self._phase_onehot()
        time_norm = np.array([min(self._time_in_phase / self.max_green, 1.0)])

        # Concatenar y asegurar valores entre 0 y 1
        obs = np.concatenate([densities, queues, phase_onehot, time_norm], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0) # Esto garantiza que nada se "dispare" fuera de rango

    def _lane_densities(self) -> np.ndarray:
        """Fracción de ocupación de cada carril: n_vehicles / capacity."""
        densities = []
        for lane in self.lanes:
            length   = self.traci.lane.getLength(lane)
            n_veh    = self.traci.lane.getLastStepVehicleNumber(lane)
            capacity = max(length / 7.5, 1)   # 7.5 m por vehículo (aprox.)
            densities.append(min(n_veh / capacity, 1.0))
        return np.array(densities, dtype=np.float32)

    def _lane_queues(self) -> np.ndarray:
        """Fracción de vehículos parados en cada carril."""
        queues = []
        for lane in self.lanes:
            length   = self.traci.lane.getLength(lane)
            n_halt   = self.traci.lane.getLastStepHaltingNumber(lane)
            capacity = max(length / 7.5, 1)
            queues.append(min(n_halt / capacity, 1.0))
        return np.array(queues, dtype=np.float32)

    def _phase_onehot(self) -> np.ndarray:
        """Vector one-hot indicando qué fase verde está activa."""
        vec = np.zeros(self.num_green_phases, dtype=np.float32)
        if self._current_phase in self.green_phases:
            idx = self.green_phases.index(self._current_phase)
            vec[idx] = 1.0
        return vec

    # ── Espacio de acción ─────────────────────────────────────────────────────

    @property
    def action_space_size(self) -> int:
        """Número de fases verdes disponibles (una acción = elegir una fase)."""
        return self.num_green_phases

    # ── Aplicar acción ────────────────────────────────────────────────────────

    def apply_action(self, action: int) -> None:
        """
        Solicita al semáforo que cambie a la fase verde `action`.

        Reglas de transición:
          1. Si ya está en esa fase Y ha cumplido min_green → no hacer nada.
          2. Si está en esa fase pero no ha cumplido min_green → tampoco.
          3. Si quiere cambiar → activar fase amarilla primero.
          4. Si ya está en amarillo → esperar a que termine.
        """
        if self._in_yellow:
            # Estamos en la transición amarilla; no hacer nada hasta que acabe
            return

        target_phase = self.green_phases[action]

        if target_phase == self._current_phase:
            # Misma fase: simplemente incrementar el contador
            self._time_in_phase += self.delta_time
            return

        if self._time_in_phase < self.min_green:
            # No ha pasado el tiempo mínimo en verde → forzar la espera
            self._time_in_phase += self.delta_time
            return

        # Cambio de fase: activar amarillo intermedio
        self._next_phase  = target_phase
        self._in_yellow   = True
        self._time_in_phase = 0

        # Construir una fase amarilla: sustituir G/g por y en la fase actual
        logic = self.traci.trafficlight.getAllProgramLogics(self.id)[0]
        current_state = logic.phases[self._current_phase].state
        yellow_state  = current_state.replace("G", "y").replace("g", "y")

        self.traci.trafficlight.setPhase(self.id, self._current_phase)
        self.traci.trafficlight.setPhaseDuration(self.id, self.yellow_time)
        # Nota: la transición real la completa `complete_yellow()` después de
        # yellow_time segundos de simulación

    def complete_yellow(self) -> None:
        """
        Llamado por el entorno tras yellow_time segundos.
        Activa la fase verde destino.
        """
        if self._in_yellow:
            self._current_phase = self._next_phase
            self._in_yellow     = False
            self._time_in_phase = 0
            self.traci.trafficlight.setPhase(self.id, self._current_phase)

    # ── Recompensa ────────────────────────────────────────────────────────────

    @property
    def reward(self) -> float:
        """
        Calcula la recompensa basada en la presión: 
        Diferencia entre vehículos en carriles de entrada y salida.
        """
        # 1. Obtener presión actual
        pressure = self._get_pressure()
        
        # 2. Recompensa = -Presión (queremos minimizar la presión)
        return -float(pressure)

    def _get_pressure(self) -> float:
        """
        Presión = sum(vehículos_en_carriles_entrada) - sum(vehículos_en_carriles_salida)
        """
        # Carriles de entrada (los que ya tienes en self.lanes)
        in_pressure = sum(self.traci.lane.getLastStepVehicleNumber(lane) for lane in self.lanes)
        
        # Carriles de salida (necesitas identificarlos)
        # Esto es un ejemplo, debes mapear qué carriles son los de salida
        out_pressure = sum(self.traci.lane.getLastStepVehicleNumber(lane) for lane in self.out_lanes)
        
        return in_pressure - out_pressure

    def _total_waiting_time(self) -> float:
        """
        Suma del tiempo de espera INSTANTÁNEO por carril.

        traci.lane.getWaitingTime(lane) devuelve la suma de waitingTime
        de todos los vehículos en ese carril en este paso de simulación.
        waitingTime de un vehículo se resetea a 0 cuando supera 0.1 m/s,
        por lo que no crece sin límite como getAccumulatedWaitingTime.
        """
        return sum(
            self.traci.lane.getWaitingTime(lane)
            for lane in self.lanes
        )

    # ── Métricas de evaluación ────────────────────────────────────────────────

    def metrics(self) -> dict:
        """Devuelve métricas locales de esta intersección para el logger."""
        halting = sum(self.traci.lane.getLastStepHaltingNumber(l) for l in self.lanes)
        waiting = sum(self.traci.lane.getWaitingTime(l)             for l in self.lanes)
        total   = sum(self.traci.lane.getLastStepVehicleNumber(l) for l in self.lanes)

        return {
            "ts_id":          self.id,
            "queue_length":   halting,
            "waiting_time":   waiting,
            "total_vehicles": total,
            "phase":          self._current_phase,
        }
