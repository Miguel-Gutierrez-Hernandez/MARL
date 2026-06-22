"""
training/logger.py
==================
Registra métricas de entrenamiento en TensorBoard (y opcionalmente WandB).

¿Para qué sirve el logger?
  Durante el entrenamiento necesitamos monitorizar:
  - Las pérdidas de las redes (¿está convergiendo?)
  - Las recompensas por episodio (¿mejora el comportamiento?)
  - Las métricas de tráfico (tiempo de espera, colas, throughput)
  - El epsilon de exploración (¿está decayendo bien?)

TensorBoard:
  Abre el dashboard con:  tensorboard --logdir logs/
  Luego en el navegador:  http://localhost:6006
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional


class Logger:
    """
    Logger unificado para TensorBoard y WandB.

    Parámetros
    ----------
    log_dir    : carpeta donde se guardan los logs
    algo       : nombre del algoritmo (para el título en TensorBoard)
    use_tb     : activar TensorBoard
    use_wandb  : activar WandB
    wandb_project : nombre del proyecto en WandB
    """

    def __init__(
        self,
        log_dir:       str,
        algo:          str  = "agent",
        use_tb:        bool = True,
        use_wandb:     bool = False,
        wandb_project: str  = "marl-traffic",
        config:        Optional[dict] = None,
    ):
        self.log_dir  = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.algo     = algo
        self._step    = 0
        self._episode = 0
        self._t_start = time.time()

        # ── TensorBoard ───────────────────────────────────────────────────────
        self.tb_writer = None
        if use_tb:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tb"))
                print(f"  [logger] TensorBoard → tensorboard --logdir {self.log_dir / 'tb'}")
            except ImportError:
                print("  [logger] TensorBoard no disponible (pip install tensorboard)")

        # ── WandB ─────────────────────────────────────────────────────────────
        self.wandb = None
        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, name=f"{algo}_{int(time.time())}",
                           config=config)
                self.wandb = wandb
                print(f"  [logger] WandB iniciado: {wandb_project}")
            except ImportError:
                print("  [logger] WandB no disponible (pip install wandb)")

        # ── CSV de respaldo (siempre activo) ──────────────────────────────────
        self._csv_path = self.log_dir / "metrics.jsonl"
        self._csv_file = open(self._csv_path, "w")

    # ── Registro de métricas ──────────────────────────────────────────────────

    def log_step(self, metrics: dict[str, float], step: Optional[int] = None):
        """Registra métricas a nivel de paso de entrenamiento."""
        t = step if step is not None else self._step
        self._write(metrics, t, prefix="train")
        self._step = t + 1

    def log_episode(self, metrics: dict[str, float], episode: Optional[int] = None):
        """Registra métricas a nivel de episodio."""
        ep = episode if episode is not None else self._episode
        self._write(metrics, ep, prefix="episode")
        self._episode = ep + 1

    def log_eval(self, metrics: dict[str, float], step: int):
        """Registra métricas de evaluación (sin exploración)."""
        self._write(metrics, step, prefix="eval")

    def _write(self, metrics: dict, step: int, prefix: str):
        """Escribe en TensorBoard, WandB y CSV."""
        elapsed = time.time() - self._t_start

        # TensorBoard
        if self.tb_writer:
            for k, v in metrics.items():
                try:
                    self.tb_writer.add_scalar(f"{prefix}/{k}", v, step)
                except Exception:
                    pass

        # WandB
        if self.wandb:
            try:
                self.wandb.log({f"{prefix}/{k}": v for k, v in metrics.items()},
                               step=step)
            except Exception:
                pass

        # JSONL (una línea JSON por entrada)
        record = {"step": step, "prefix": prefix, "elapsed": elapsed, **metrics}
        self._csv_file.write(json.dumps(record) + "\n")
        self._csv_file.flush()

    # ── Cierre ────────────────────────────────────────────────────────────────

    def close(self):
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb:
            self.wandb.finish()
        self._csv_file.close()

    # ── Contexto ──────────────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
