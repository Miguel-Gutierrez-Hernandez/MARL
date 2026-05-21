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
        self._last_pressure   : float = 0.0  # Para calcular la recompensa delta
        self._phase_changed   : bool = False  # Detectar cambio de fase (para penalizar flickering)

        # Caché de la lógica de semáforos (se llena en _setup)
        self.green_phases     : list[int] = []
        self._phase_states    : list[str] = []
        self.lanes            : list[str] = []
        self.out_lanes        : list[str] = []
        self.num_green_phases : int = 0
        self._phase_count     : int = 0
        self._yellow_state    : str | None = None
        self._phase_changed   : bool = False

        self._setup()

    # ── Inicialización ─────────────────────────────────────────────────────────

    def _setup(self) -> None:
        """Lee de SUMO la lógica del semáforo y los carriles que controla."""
        # Obtener la lógica de semáforo del nodo
        logic = self.traci.trafficlight.getAllProgramLogics(self.id)[0]
        all_phases = logic.phases
        self._phase_count = len(all_phases)

        # Identificar fases verdes (las que contienen 'G' o 'g' pero no 'y'/'Y')
        self._phase_states = [
            p.state for p in all_phases
            if ("G" in p.state or "g" in p.state)
            and "y" not in p.state.lower()
        ]

        # Carriles controlados por este semáforo
        controlled_links = self.traci.trafficlight.getControlledLinks(self.id)
        seen_in = set()
        seen_out = set()
        for link_list in controlled_links:
            for link in link_list:
                if not link:
                    continue
                if link[0] and link[0] not in seen_in:
                    self.lanes.append(link[0])   # link[0] = carril de entrada
                    seen_in.add(link[0])
                if link[1] and link[1] not in seen_out:
                    self.out_lanes.append(link[1])   # link[1] = carril de salida
                    seen_out.add(link[1])

        # Si SUMO no definió fases útiles, construimos fases manuales por dirección
        if len(self._phase_states) <= 1:
            self._phase_states = self._build_default_phase_states(controlled_links)
            phases = [
                self.traci.trafficlight.Phase(self.max_green, state)
                for state in self._phase_states
            ]
            custom_logic = self.traci.trafficlight.Logic(
                "0",
                logic.type,
                0,
                phases=phases,
            )
            self.traci.trafficlight.setProgramLogic(self.id, custom_logic)
            self.traci.trafficlight.setProgram(self.id, "0")
            self._phase_count = len(phases)

        self.num_green_phases = len(self._phase_states)
        self.green_phases = list(range(self.num_green_phases))

        # Número de carriles por fase (para el espacio de observación)
        self._num_lanes = len(self.lanes)

        # Fase inicial
        self._current_phase = 0
        self.traci.trafficlight.setRedYellowGreenState(self.id, self._phase_states[0])

    def _build_default_phase_states(self, controlled_links: list[list[tuple[str, str, str]]]) -> list[str]:
        """Crea fases manuales por dirección cuando SUMO solo genera una fase estática."""
        # Ordenamos los enlaces entrantes por dirección y agruparlos.
        groups: dict[str, list[int]] = {}
        order: list[str] = []
        for signal_idx, link_list in enumerate(controlled_links):
            if not link_list or not link_list[0] or not link_list[0][0]:
                continue
            in_edge = link_list[0][0].split("_")[0]
            if in_edge not in groups:
                groups[in_edge] = []
                order.append(in_edge)
            groups[in_edge].append(signal_idx)

        # Si no hay suficientes grupos para construir varias fases, conservar el estado actual.
        if len(order) <= 1:
            base_state = self._phase_states[0] if self._phase_states else ""
            return [base_state or "G" * len(controlled_links)]

        signal_count = len(self._phase_states[0]) if self._phase_states else len(controlled_links)
        phase_states = []
        for in_edge in order:
            indices = set(groups[in_edge])
            state = "".join(
                "G" if idx in indices else "r"
                for idx in range(signal_count)
            )
            phase_states.append(state)

        return phase_states

    # ── Espacio de observación ─────────────────────────────────────────────────

    @property
    def observation_space_size(self) -> int:
        """
        Tamaño del vector de observación:
          - densidad por carril      (num_lanes valores en [0,1])
          - cola normalizada          (num_lanes valores en [0,1])
          - espera normalizada        (num_lanes valores en [0,1])
          - fase actual (one-hot)    (num_green_phases valores en {0,1})
          - tiempo en fase normaliz. (1 valor en [0,1])
        """
        return self._num_lanes * 3 + self.num_green_phases + 1

    @property
    def observation(self) -> np.ndarray:
        densities = self._lane_densities()
        queues = self._lane_queues()
        waits = self._lane_waits()
        phase_onehot = self._phase_onehot()
        time_norm = np.array([min(self._time_in_phase / self.max_green, 1.0)])

        # Concatenar y asegurar valores entre 0 y 1
        obs = np.concatenate([densities, queues, waits, phase_onehot, time_norm], dtype=np.float32)
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

    def _lane_waits(self) -> np.ndarray:
        """Tiempo de espera medio normalizado en cada carril."""
        waits = []
        for lane in self.lanes:
            w = self.traci.lane.getWaitingTime(lane)
            n = max(self.traci.lane.getLastStepVehicleNumber(lane), 1)
            waits.append(min((w / n) / 100.0, 1.0)) # Normalizado a 100 segundos
        return np.array(waits, dtype=np.float32)

    def _phase_onehot(self) -> np.ndarray:
        """Vector one-hot indicando qué fase verde está activa."""
        vec = np.zeros(self.num_green_phases, dtype=np.float32)
        if 0 <= self._current_phase < self.num_green_phases:
            vec[self._current_phase] = 1.0
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
        self._phase_changed = False
        if self._in_yellow:
            # Estamos en la transición amarilla; mantener el estado mientras la espera
            if self._yellow_state is not None:
                self.traci.trafficlight.setRedYellowGreenState(self.id, self._yellow_state)
            return

        target_phase = self.green_phases[action]

        if target_phase == self._current_phase:
            # Misma fase: simplemente incrementar el contador
            self._time_in_phase += self.delta_time
            self._phase_changed = False
            return

        if self._time_in_phase < self.min_green:
            # No ha pasado el tiempo mínimo en verde → forzar la espera
            self._time_in_phase += self.delta_time
            self._phase_changed = False
            return

        # Cambio de fase: activar amarillo intermedio
        self._phase_changed = True
        self._next_phase    = target_phase
        self._in_yellow     = True
        self._time_in_phase = 0
        self._phase_changed = True  # Marcar que se intenta cambiar de fase

        # Construir una fase amarilla: sustituir G/g por y en la fase actual
        current_state = self._phase_states[self._current_phase]
        yellow_state  = current_state.replace("G", "y").replace("g", "y")
        self._yellow_state = yellow_state

        self.traci.trafficlight.setRedYellowGreenState(self.id, yellow_state)
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
            self._yellow_state  = None
            self.traci.trafficlight.setRedYellowGreenState(
                self.id,
                self._phase_states[self._current_phase],
            )

    # ── Recompensa ────────────────────────────────────────────────────────────

    @property
    def reward(self) -> float:
        """
        Recompensa basada solo en delta-pressure normalizado.
        
        - Señal principal: delta-presión (reducción de congestión)
        - Penalización: -1.0 si el agente cambió de fase (anti-flickering)
        """
        pressure = self._get_pressure()
        delta_pressure = float(self._last_pressure - pressure)
        pressure_norm = delta_pressure / max(abs(self._last_pressure) + abs(pressure), 1.0)

        # Penalización por cambio de fase (flickering)
        phase_change_penalty = -1.0 if self._phase_changed else 0.0
        
        reward = pressure_norm + phase_change_penalty
        
        self._last_pressure = pressure
        self._phase_changed = False  # Resetear el flag después de usar
        return reward

    def _get_pressure(self) -> float:
        """
        Presión local = vehículos en carriles de entrada - vehículos en carriles de salida.
        """
        in_pressure = sum(
            self.traci.lane.getLastStepVehicleNumber(lane)
            for lane in self.lanes
        )
        out_pressure = sum(
            self.traci.lane.getLastStepVehicleNumber(lane)
            for lane in self.out_lanes
        )
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
    def _total_queue_length(self) -> float:
        """Número total de vehículos detenidos en los carriles de entrada."""
        return sum(
            self.traci.lane.getLastStepHaltingNumber(lane)
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
