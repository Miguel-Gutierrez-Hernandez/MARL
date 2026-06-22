"""
evaluation/visualizer.py
=========================
Genera todas las gráficas del TFM:

  1. Curvas de entrenamiento   (recompensa / tiempo de espera por episodio)
  2. Comparativa de algoritmos (barras con IC95%)
  3. Heatmap de escalabilidad  (métrica × red × algoritmo)
  4. Convergencia de epsilon   (solo para algoritmos value-based)
  5. Gráfica de equidad        (Jain's fairness index)

Uso:
    from evaluation.visualizer import plot_training_curves, plot_comparison

    plot_training_curves("logs/iql_2x2_moderate/run_1/metrics.jsonl",
                         save_path="results/iql_training.png")

    plot_comparison(results_dict, network="2x2", demand="moderate",
                    save_dir="results/")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


# ── Estilo global ─────────────────────────────────────────────────────────────
ALGO_COLORS = {
    "iql":     "#4C72B0",
    "qmix":    "#DD8452",
    "mappo":   "#55A868",
    "maddpg":  "#C44E52",
    "commnet": "#8172B2",
}
ALGO_LABELS = {
    "iql":     "IQL",
    "qmix":    "QMIX",
    "mappo":   "MAPPO",
    "maddpg":  "MADDPG",
    "commnet": "CommNet/TarMAC",
}

def _set_style():
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi":      150,
        "savefig.dpi":     300,
        "savefig.bbox":    "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ── 1. Curvas de entrenamiento ────────────────────────────────────────────────

def plot_training_curves(
    jsonl_path: str,
    metrics:    list[str] = ("mean_reward", "total_waiting"),
    smooth:     int       = 10,
    save_path:  Optional[str] = None,
) -> None:
    """
    Curvas de entrenamiento leídas del archivo metrics.jsonl de una run.

    Parámetros
    ----------
    smooth : ventana de suavizado (media móvil)
    """
    _set_style()
    records = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("prefix") == "episode":
                    records.append(r)
            except json.JSONDecodeError:
                continue

    if not records:
        print("[visualizer] No hay datos de episodio en", jsonl_path)
        return

    df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        if metric not in df.columns:
            continue
        vals   = df[metric].values
        smooth_vals = pd.Series(vals).rolling(smooth, min_periods=1).mean().values
        episodes    = np.arange(len(vals))

        ax.plot(episodes, vals,        alpha=0.3, color="steelblue", linewidth=0.8)
        ax.plot(episodes, smooth_vals, color="steelblue", linewidth=2.0,
                label=f"Media móvil ({smooth})")
        ax.set_xlabel("Episodio")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title())
        ax.legend(fontsize=9)

    fig.suptitle(f"Curvas de entrenamiento — {Path(jsonl_path).parent.name}",
                 fontsize=12, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"  [visualizer] Guardado: {save_path}")
    else:
        plt.show()
    plt.close()


# ── 2. Comparativa de algoritmos ──────────────────────────────────────────────

def plot_comparison(
    results:   list[dict],
    network:   str,
    demand:    str,
    metrics:   list[str] = ("mean_waiting_time", "mean_throughput",
                             "mean_queue_length", "fairness_index"),
    save_dir:  Optional[str] = None,
) -> None:
    """
    Barras comparativas para cada métrica, con todos los algoritmos.

    Parámetros
    ----------
    results : lista de dicts (uno por algoritmo), cada uno con claves
              "algo", "mean_waiting_time", "std_waiting_time", etc.
    """
    _set_style()

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        algos  = [r["algo"] for r in results]
        values = [r.get(metric, 0) for r in results]
        errors = [r.get(f"std_{metric.replace('mean_', '')}", 0) for r in results]
        colors = [ALGO_COLORS.get(a, "#888888") for a in algos]
        labels = [ALGO_LABELS.get(a, a.upper()) for a in algos]

        bars = ax.bar(labels, values, yerr=errors, capsize=5,
                      color=colors, alpha=0.85, edgecolor="white", linewidth=0.5)

        # Anotar valores sobre las barras
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=9)

        ax.set_title(metric.replace("_", " ").replace("mean ", "").title())
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(metric.split("_")[-1].title())

        # Para fairness: eje limitado [0, 1]
        if "fairness" in metric:
            ax.set_ylim(0, 1.1)

    fig.suptitle(f"Comparativa de algoritmos — Red {network}, Demanda {demand}",
                 fontsize=13, y=1.02)
    plt.tight_layout()

    if save_dir:
        path = Path(save_dir) / f"comparison_{network}_{demand}.png"
        plt.savefig(str(path))
        print(f"  [visualizer] Guardado: {path}")
    else:
        plt.show()
    plt.close()


# ── 3. Heatmap de escalabilidad ───────────────────────────────────────────────

def plot_scalability_heatmap(
    data:      dict[str, dict[str, float]],
    metric:    str = "mean_waiting_time",
    save_path: Optional[str] = None,
) -> None:
    """
    Heatmap: algoritmos (filas) × topologías de red (columnas).

    Parámetros
    ----------
    data : dict[algo → dict[network → valor]]
           Ejemplo: {"iql": {"2x2": 45.2, "3x3": 78.1, "4x4": 110.3}, ...}
    """
    _set_style()

    algos    = list(data.keys())
    networks = ["2x2", "3x3", "4x4"]

    matrix = np.array([
        [data[a].get(n, np.nan) for n in networks]
        for a in algos
    ])

    fig, ax = plt.subplots(figsize=(6, len(algos) * 1.0 + 1.5))
    cmap    = "RdYlGn_r" if "waiting" in metric or "queue" in metric else "RdYlGn"

    im = sns.heatmap(
        matrix,
        ax          = ax,
        annot       = True,
        fmt         = ".1f",
        xticklabels = networks,
        yticklabels = [ALGO_LABELS.get(a, a.upper()) for a in algos],
        cmap        = cmap,
        linewidths  = 0.5,
        cbar_kws    = {"label": metric.replace("_", " ").title()},
    )

    ax.set_title(f"Escalabilidad — {metric.replace('_', ' ').title()}", pad=12)
    ax.set_xlabel("Topología de red")
    ax.set_ylabel("Algoritmo")

    if save_path:
        plt.savefig(save_path)
        print(f"  [visualizer] Guardado: {save_path}")
    else:
        plt.show()
    plt.close()


# ── 4. Curva de convergencia multi-run ───────────────────────────────────────

def plot_multi_run_convergence(
    runs_data: dict[str, list[list[float]]],
    metric:    str  = "mean_waiting_time",
    smooth:    int  = 10,
    save_path: Optional[str] = None,
) -> None:
    """
    Media ± desviación entre runs para cada algoritmo.

    Parámetros
    ----------
    runs_data : dict[algo → lista de series (una por run)]
    """
    _set_style()
    fig, ax = plt.subplots(figsize=(9, 5))

    for algo, runs in runs_data.items():
        color = ALGO_COLORS.get(algo, "#888888")
        label = ALGO_LABELS.get(algo, algo.upper())

        # Alinear longitudes
        min_len = min(len(r) for r in runs)
        arr = np.array([r[:min_len] for r in runs], dtype=float)

        # Suavizado por run
        smooth_arr = np.array([
            pd.Series(arr[i]).rolling(smooth, min_periods=1).mean().values
            for i in range(len(runs))
        ])

        mean = smooth_arr.mean(axis=0)
        std  = smooth_arr.std(axis=0)
        x    = np.arange(min_len)

        ax.plot(x, mean, color=color, linewidth=2.0, label=label)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel("Episodio")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(f"Convergencia — {metric.replace('_', ' ').title()}")
    ax.legend(loc="upper right")

    if save_path:
        plt.savefig(save_path)
        print(f"  [visualizer] Guardado: {save_path}")
    else:
        plt.show()
    plt.close()


# ── __init__ ──────────────────────────────────────────────────────────────────

def __getattr__(name):
    # Lazy import guard para evitar errores si matplotlib no está disponible
    raise AttributeError(f"No existe '{name}' en evaluation.visualizer")
