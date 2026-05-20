"""
evaluation/statistical.py
==========================
Tests estadísticos para comparar algoritmos MARL.

¿Por qué tests no paramétricos?
  En RL, las métricas raramente siguen una distribución normal.
  Mann-Whitney U es más robusto y no asume normalidad.

Funciones principales:
  - compare_algorithms()  : tabla comparativa con intervalos de confianza
  - pairwise_tests()      : tests Mann-Whitney entre todos los pares
  - summary_table()       : tabla LaTeX para el TFM

Uso típico:
  results = {
    "iql":    [wt_run1, wt_run2, wt_run3, wt_run4, wt_run5],
    "qmix":   [...],
    ...
  }
  df = compare_algorithms(results)
  print(df.to_string())
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ── Comparación principal ─────────────────────────────────────────────────────

def compare_algorithms(
    results:    dict[str, list[float]],
    metric:     str  = "mean_waiting_time",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Genera una tabla con media ± IC95% para cada algoritmo.

    Parámetros
    ----------
    results    : dict[algo → lista de valores por run]
    metric     : nombre de la métrica (para el título)
    confidence : nivel de confianza del intervalo

    Returns
    -------
    DataFrame con columnas: algo, mean, std, ci_low, ci_high, n_runs
    """
    rows = []
    alpha = 1 - confidence

    for algo, values in results.items():
        arr = np.array(values, dtype=float)
        n   = len(arr)
        m   = arr.mean()
        s   = arr.std(ddof=1) if n > 1 else 0.0

        # Intervalo de confianza de Student (t-distribution)
        if n > 1:
            t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
            margin = t_crit * s / np.sqrt(n)
        else:
            margin = 0.0

        rows.append({
            "algoritmo": algo.upper(),
            "media":     round(m,          4),
            "std":       round(s,          4),
            "ic_low":    round(m - margin, 4),
            "ic_high":   round(m + margin, 4),
            "n_runs":    n,
        })

    df = pd.DataFrame(rows).sort_values("media")
    return df


# ── Tests de Mann-Whitney por pares ───────────────────────────────────────────

def pairwise_tests(
    results: dict[str, list[float]],
    alpha:   float = 0.05,
) -> pd.DataFrame:
    """
    Ejecuta el test de Mann-Whitney U para todos los pares de algoritmos.

    El test comprueba si la distribución de A es significativamente
    diferente a la de B (hipótesis alternativa: A < B en mediana).

    Returns
    -------
    DataFrame con columnas: algo_a, algo_b, U_stat, p_value, significativo
    """
    algos = list(results.keys())
    rows  = []

    for i in range(len(algos)):
        for j in range(i + 1, len(algos)):
            a, b = algos[i], algos[j]
            u_stat, p_val = stats.mannwhitneyu(
                results[a], results[b],
                alternative="two-sided",
            )
            rows.append({
                "algo_a":       a.upper(),
                "algo_b":       b.upper(),
                "U_stat":       round(u_stat, 2),
                "p_value":      round(p_val,  4),
                "significativo": "✓" if p_val < alpha else "✗",
            })

    return pd.DataFrame(rows)


# ── Tabla LaTeX para el TFM ───────────────────────────────────────────────────

def summary_table_latex(
    results_by_metric: dict[str, dict[str, list[float]]],
) -> str:
    """
    Genera una tabla LaTeX lista para pegar en el TFM.

    Parámetros
    ----------
    results_by_metric : dict[métrica → dict[algo → lista de valores]]

    Ejemplo de entrada:
        {
          "waiting_time": {"iql": [...], "qmix": [...], ...},
          "throughput":   {"iql": [...], "qmix": [...], ...},
        }
    """
    algos   = list(next(iter(results_by_metric.values())).keys())
    metrics = list(results_by_metric.keys())

    header_cols = " & ".join([f"\\textbf{{{m}}}" for m in metrics])
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\begin{{tabular}}{{l{'c' * len(metrics)}}}",
        "\\toprule",
        f"\\textbf{{Algoritmo}} & {header_cols} \\\\",
        "\\midrule",
    ]

    for algo in algos:
        cells = []
        for metric in metrics:
            vals = results_by_metric[metric][algo]
            arr  = np.array(vals)
            cell = f"{arr.mean():.2f} $\\pm$ {arr.std(ddof=1):.2f}"
            cells.append(cell)
        lines.append(f"{algo.upper()} & {' & '.join(cells)} \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Comparativa de algoritmos MARL (media $\\pm$ std, 5 runs)}",
        "\\label{tab:results}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ── Cargar resultados desde logs/ ────────────────────────────────────────────

def load_eval_results(
    logs_dir:  str,
    algos:     list[str],
    network:   str,
    demand:    str,
    metric:    str = "mean_waiting_time",
) -> dict[str, list[float]]:
    """
    Lee los archivos metrics.jsonl de cada run y extrae los valores
    de evaluación para la métrica indicada.

    Returns
    -------
    dict[algo → lista con el valor final de evaluación de cada run]
    """
    results = {}
    logs    = Path(logs_dir)

    for algo in algos:
        algo_dir = logs / f"{algo}_{network}_{demand}"
        if not algo_dir.exists():
            print(f"  [!] No encontrado: {algo_dir}")
            continue

        run_values = []
        for run_dir in sorted(algo_dir.glob("run_*")):
            jsonl = run_dir / "metrics.jsonl"
            if not jsonl.exists():
                continue

            # Leer última métrica de evaluación del archivo
            last_eval = None
            with open(jsonl) as f:
                for line in f:
                    try:
                        record = json.loads(line)
                        if record.get("prefix") == "eval" and metric in record:
                            last_eval = record[metric]
                    except json.JSONDecodeError:
                        continue

            if last_eval is not None:
                run_values.append(last_eval)

        if run_values:
            results[algo] = run_values

    return results
