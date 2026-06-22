"""
train.py — Script principal de entrenamiento
=============================================
Uso:
    python train.py --algo iql --network 2x2 --demand moderate
    python train.py --algo qmix --network 3x3 --runs 5
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml


# ── Argumentos de línea de comandos ─────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenamiento MARL para semáforos")

    parser.add_argument("--algo", type=str, default="iql",
                        choices=["iql", "qmix", "mappo", "maddpg", "commnet"],
                        help="Algoritmo MARL a entrenar")

    parser.add_argument("--network", type=str, default="2x2",
                        choices=["2x2", "3x3", "4x4"],
                        help="Topología de la red viaria")

    parser.add_argument("--demand", type=str, default="moderate",
                        choices=["low", "moderate", "saturated"],
                        help="Nivel de demanda de tráfico")

    parser.add_argument("--runs", type=int, default=1,
                        help="Número de ejecuciones con semillas distintas")

    parser.add_argument("--gui", action="store_true",
                        help="Abrir interfaz gráfica de SUMO (solo para depuración)")

    parser.add_argument("--config", type=str, default=None,
                        help="Ruta a un archivo .yaml de configuración personalizado")

    parser.add_argument("--timesteps", type=int, default=None,
                        help="Sobreescribir total_timesteps de la config")

    parser.add_argument("--device", type=str, default=None,
                        choices=["cpu", "cuda", "mps"],
                        help="Dispositivo de cómputo")

    return parser.parse_args()


# ── Carga de configuración ───────────────────────────────────────────────────

def load_config(algo: str, override_path: str = None) -> dict:
    """Carga default.yaml y lo fusiona con el yaml del algoritmo."""
    config_dir = Path(__file__).parent / "config"

    with open(config_dir / "default.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    algo_config_path = config_dir / f"{algo}.yaml"
    if algo_config_path.exists():
        with open(algo_config_path, "r", encoding="utf-8") as f:
            algo_config = yaml.safe_load(f)
        config.update(algo_config)

    # Configuración personalizada (tiene máxima prioridad)
    if override_path:
        with open(override_path, "r", encoding="utf-8") as f:
            override = yaml.safe_load(f)
        config.update(override)

    return config


# ── Reproducibilidad ─────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Creación del entorno ─────────────────────────────────────────────────────

def make_env(config: dict, use_gui: bool = False):
    """
    Crea el entorno multi-agente SUMO.
    Importamos aquí para que el script no falle si SUMO no está instalado
    al consultar el --help.
    """
    from env.sumo_env import MultiAgentSumoEnv  # Se implementará en el Paso 2

    return MultiAgentSumoEnv(
        network=config["env"]["network"],
        demand=config["env"]["demand"],
        num_seconds=config["env"]["num_seconds"],
        delta_time=config["env"]["delta_time"],
        yellow_time=config["env"]["yellow_time"],
        min_green=config["env"]["min_green"],
        max_green=config["env"]["max_green"],
        use_gui=use_gui,
    )


# ── Creación del agente ──────────────────────────────────────────────────────

def make_agent(algo: str, env, config: dict, device: str):
    """Instancia el algoritmo MARL correspondiente."""
    # Cada módulo se implementará en el Paso 3
    if algo == "iql":
        from agents.iql import IQLAgent
        return IQLAgent(env, config, device)
    elif algo == "qmix":
        from agents.qmix import QMIXAgent
        return QMIXAgent(env, config, device)
    elif algo == "mappo":
        from agents.mappo import MAPPOAgent
        return MAPPOAgent(env, config, device)
    elif algo == "maddpg":
        from agents.maddpg import MADDPGAgent
        return MADDPGAgent(env, config, device)
    elif algo == "commnet":
        from agents.commnet import CommNetAgent
        return CommNetAgent(env, config, device)
    else:
        raise ValueError(f"Algoritmo desconocido: {algo}")


# ── Bucle principal de entrenamiento ─────────────────────────────────────────

def train_one_run(algo: str, config: dict, seed: int, run_id: int,
                  use_gui: bool, device: str) -> dict:
    """
    Ejecuta un entrenamiento completo y devuelve las métricas finales.
    """
    set_seed(seed)

    print(f"\n{'═'*60}")
    print(f"  Algoritmo : {algo.upper()}")
    print(f"  Red       : {config['env']['network']}")
    print(f"  Demanda   : {config['env']['demand']}")
    print(f"  Semilla   : {seed}  (run {run_id})")
    print(f"  Dispositivo: {device}")
    print(f"{'═'*60}\n")

    # Directorio de salida para este experimento
    run_dir = Path("logs") / f"{algo}_{config['env']['network']}_{config['env']['demand']}" / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Guardar configuración usada
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Entorno y agente
    env = make_env(config, use_gui=use_gui)
    env.reset(seed=seed)
    agent = make_agent(algo, env, config, device)

    # Importar trainer (se implementará en el Paso 4)
    from training.trainer import Trainer
    trainer = Trainer(
        env=env,
        agent=agent,
        config=config,
        log_dir=run_dir,
        seed=seed,
    )

    # ¡A entrenar!
    t_start = time.time()
    metrics = trainer.train()
    elapsed = time.time() - t_start

    # Guardar resultados finales
    print(f"  Recompensa media final    : {metrics.get('mean_reward', '?')}")
    
    waiting_time = metrics.get('mean_waiting_time', '?')
    if isinstance(waiting_time, (int, float)):
        print(f"  Tiempo espera medio final : {waiting_time:.2f} s")
    else:
        print(f"  Tiempo espera medio final : {waiting_time} (No evaluado)")

    env.close()
    return metrics


# ── Punto de entrada ─────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Configuración
    config = load_config(args.algo, args.config)

    # Sobreescrituras desde CLI
    config["env"]["network"] = args.network
    config["env"]["demand"] = args.demand
    config["env"]["use_gui"] = args.gui

    if args.timesteps:
        config["training"]["total_timesteps"] = args.timesteps

    device = args.device or config["training"].get("device", "cpu")

    # Ejecutar N repeticiones con semillas distintas
    seeds = [config["env"]["seed"] + i for i in range(args.runs)]
    all_metrics = []

    for run_id, seed in enumerate(seeds, start=1):
        metrics = train_one_run(
            algo=args.algo,
            config=config,
            seed=seed,
            run_id=run_id,
            use_gui=args.gui,
            device=device,
        )
        all_metrics.append(metrics)

    # Resumen si hay más de una run
    if args.runs > 1:
        print(f"\n{'═'*60}")
        print(f"  RESUMEN — {args.runs} runs completadas")
        print(f"{'═'*60}")
        wt = [m.get("mean_waiting_time", 0) for m in all_metrics]
        print(f"  Tiempo espera: {np.mean(wt):.2f} ± {np.std(wt):.2f} s")

    print("\n✓ Entrenamiento finalizado. Resultados en logs/")


if __name__ == "__main__":
    main()
