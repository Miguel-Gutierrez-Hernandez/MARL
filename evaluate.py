"""
evaluate.py — Evaluación de modelos entrenados
===============================================
Uso:
    # Evaluar un modelo concreto
    python evaluate.py --checkpoint logs/iql_2x2_moderate/run_1/best_model.pt

    # Comparar todos los algoritmos en una red
    python evaluate.py --compare --network 3x3 --demand moderate

    # Ver la simulación en el GUI de SUMO
    python evaluate.py --checkpoint logs/qmix_2x2_moderate/run_1/best_model.pt --gui
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


# ── Argumentos ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluación de modelos MARL")

    # Modo 1: evaluar un único checkpoint
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Ruta al archivo .pt del modelo a evaluar")

    # Modo 2: comparar todos los algoritmos
    parser.add_argument("--compare", action="store_true",
                        help="Comparar todos los algoritmos disponibles en logs/")

    parser.add_argument("--network", type=str, default="2x2",
                        choices=["2x2", "3x3", "4x4"])

    parser.add_argument("--demand", type=str, default="moderate",
                        choices=["low", "moderate", "saturated"])

    parser.add_argument("--episodes", type=int, default=10,
                        help="Número de episodios de evaluación")

    parser.add_argument("--gui", action="store_true",
                        help="Visualizar en el GUI de SUMO")

    parser.add_argument("--save_plots", action="store_true",
                        help="Guardar gráficas en results/")

    return parser.parse_args()


# ── Evaluación de un modelo ──────────────────────────────────────────────────

def evaluate_checkpoint(checkpoint_path: str, episodes: int,
                         use_gui: bool) -> dict:
    """
    Carga un modelo guardado y lo evalúa durante N episodios.
    Devuelve un diccionario con todas las métricas.
    """
    checkpoint_path = Path(checkpoint_path)
    run_dir = checkpoint_path.parent

    # Cargar configuración original del experimento
    with open(run_dir / "config.yaml") as f:
        config = yaml.safe_load(f)

    algo = config.get("algorithm", "iql")
    config["env"]["use_gui"] = use_gui

    print(f"\nEvaluando: {algo.upper()} — {config['env']['network']} — {config['env']['demand']}")
    print(f"Checkpoint: {checkpoint_path}")

    # Importar y construir entorno + agente
    from train import make_env, make_agent
    device = config["training"].get("device", "cpu")
    env = make_env(config, use_gui=use_gui)
    
    # Inicializamos el entorno para que cargue los agentes en memoria antes de crear la red
    env.reset()
    
    agent = make_agent(algo, env, config, device)
    agent.load(str(checkpoint_path))
    agent.eval()  # Modo evaluación: sin exploración

    # Importar métricas (se implementará en el Paso 5)
    from evaluation.metrics import EpisodeMetrics

    episode_results = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_metrics = EpisodeMetrics()

        while not done:
            actions = agent.act(obs, explore=False)
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            ep_metrics.update(info)
            done = all(terminated.values()) or all(truncated.values())
            obs = next_obs

        ep_summary = ep_metrics.summarize()
        episode_results.append(ep_summary)
        print(f"  Ep {ep+1:2d}/{episodes}  "
              f"espera={ep_summary['mean_waiting_time']:.1f}s  "
              f"cola={ep_summary['mean_queue_length']:.1f}  "
              f"throughput={ep_summary['throughput']:.0f} veh/h")

    env.close()

    # Agregar resultados
    summary = {
        "algo": algo,
        "network": config["env"]["network"],
        "demand": config["env"]["demand"],
        "mean_waiting_time":  np.mean([r["mean_waiting_time"]  for r in episode_results]),
        "std_waiting_time":   np.std ([r["mean_waiting_time"]  for r in episode_results]),
        "mean_queue_length":  np.mean([r["mean_queue_length"]  for r in episode_results]),
        "mean_throughput":    np.mean([r["throughput"]         for r in episode_results]),
        "mean_stops":         np.mean([r["mean_stops"]         for r in episode_results]),
        "fairness_index":     np.mean([r["fairness_index"]     for r in episode_results]),
    }

    print(f"\n  ── Resumen ({episodes} episodios) ──")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k:<25}: {v:.4f}")

    return summary


# ── Modo comparación ─────────────────────────────────────────────────────────

def compare_all(network: str, demand: str, episodes: int,
                use_gui: bool, save_plots: bool):
    """
    Busca en logs/ todos los algoritmos entrenados para la red y demanda dadas,
    evalúa cada uno y genera la tabla/gráfica comparativa.
    """
    algos = ["iql", "qmix", "mappo", "maddpg", "commnet"]
    results = []

    for algo in algos:
        run_dirs = sorted(Path("logs").glob(f"{algo}_{network}_{demand}/run_*"))
        if not run_dirs:
            print(f"  [!] No se encontraron runs para {algo}_{network}_{demand}")
            continue

        # Evaluar la última run disponible (o la mejor)
        checkpoint = run_dirs[-1] / "best_model.pt"
        if not checkpoint.exists():
            checkpoint = run_dirs[-1] / "final_model.pt"
        if not checkpoint.exists():
            print(f"  [!] No hay checkpoint para {algo}")
            continue

        summary = evaluate_checkpoint(str(checkpoint), episodes, use_gui)
        results.append(summary)

    if not results:
        print("\n[!] No se encontró ningún modelo entrenado. Ejecuta train.py primero.")
        return

    # Imprimir tabla comparativa
    print(f"\n{'═'*70}")
    print(f"{'COMPARATIVA':^70}")
    print(f"  Red: {network}   Demanda: {demand}")
    print(f"{'═'*70}")
    header = f"{'Algoritmo':<12} {'Espera (s)':>12} {'Cola':>8} {'Throughput':>12} {'Paradas':>8} {'Equidad':>8}"
    print(header)
    print("─" * 70)
    for r in results:
        print(f"{r['algo'].upper():<12} "
              f"{r['mean_waiting_time']:>12.2f} "
              f"{r['mean_queue_length']:>8.2f} "
              f"{r['mean_throughput']:>12.1f} "
              f"{r['mean_stops']:>8.2f} "
              f"{r['fairness_index']:>8.4f}")
    print("═" * 70)

    # Guardar resultados en JSON
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"comparison_{network}_{demand}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Resultados guardados en {out_path}")

    # Gráficas
    if save_plots:
        from evaluation.visualizer import plot_comparison
        plot_comparison(results, network, demand, save_dir=results_dir)
        print(f"✓ Gráficas guardadas en {results_dir}/")


# ── Punto de entrada ─────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.compare:
        compare_all(
            network=args.network,
            demand=args.demand,
            episodes=args.episodes,
            use_gui=args.gui,
            save_plots=args.save_plots,
        )
    elif args.checkpoint:
        evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            episodes=args.episodes,
            use_gui=args.gui,
        )
    else:
        print("Especifica --checkpoint <ruta> o usa --compare")
        print("Ejecuta  python evaluate.py --help  para más información")


if __name__ == "__main__":
    main()
