# MARL Traffic — TFM MUIA
**Multi-Agent Reinforcement Learning para coordinación de semáforos**

---

## 1. Requisitos previos

| Herramienta | Versión mínima | Notas |
|-------------|---------------|-------|
| Python      | 3.10+         | Recomendado: 3.11 |
| pip         | 23+           | `pip install --upgrade pip` |
| Git         | cualquiera    | Para clonar el repo |

> **SUMO no hace falta instalarlo manualmente.** El paquete `eclipse-sumo` lo instala todo vía pip (binarios incluidos).

---

## 2. Instalación (una sola vez)

### 2.1 Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/marl_traffic.git
cd marl_traffic
```

### 2.2 Crear entorno virtual
```bash
# Crear
python -m venv venv

# Activar (Linux / macOS)
source venv/bin/activate

# Activar (Windows PowerShell)
.\venv\Scripts\Activate.ps1
```

### 2.3 Instalar dependencias
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Esto instala **SUMO automáticamente** (≈ 500 MB, paciencia la primera vez).

### 2.4 Verificar SUMO
```bash
python -c "import sumo; print('SUMO OK:', sumo.__version__)"
```
Si aparece la versión, todo está bien. Si falla, prueba:
```bash
pip install eclipse-sumo --force-reinstall
```

### 2.5 Variable de entorno SUMO_HOME (solo si ves errores de rutas)
```bash
# Linux / macOS — añade esto a tu ~/.bashrc o ~/.zshrc
export SUMO_HOME=$(python -c "import sumo; import os; print(os.path.dirname(sumo.__file__))")

# Windows PowerShell
$env:SUMO_HOME = python -c "import sumo; import os; print(os.path.dirname(sumo.__file__))"
```

---

## 3. Estructura del proyecto

```
marl_traffic/
│
├── config/               ← Hiperparámetros de cada algoritmo
│   ├── default.yaml
│   ├── iql.yaml
│   ├── qmix.yaml
│   ├── mappo.yaml
│   ├── maddpg.yaml
│   └── commnet.yaml
│
├── networks/             ← Redes viarias SUMO (2×2, 3×3, 4×4)  [Paso 2]
│
├── env/                  ← Entorno Gymnasium sobre SUMO          [Paso 2]
│   ├── sumo_env.py
│   ├── traffic_signal.py
│   └── demand.py
│
├── agents/               ← Algoritmos MARL                       [Paso 3]
│   ├── base_agent.py
│   ├── iql.py
│   ├── qmix.py
│   ├── mappo.py
│   ├── maddpg.py
│   └── commnet.py
│
├── networks_nn/          ← Arquitecturas de red neuronal         [Paso 3]
│
├── training/             ← Bucle de entrenamiento y buffer       [Paso 4]
│   ├── trainer.py
│   ├── replay_buffer.py
│   └── logger.py
│
├── evaluation/           ← Métricas, tests estadísticos, gráficas [Paso 5]
│   ├── metrics.py
│   ├── statistical.py
│   └── visualizer.py
│
├── train.py              ← Script principal de entrenamiento
├── evaluate.py           ← Script de evaluación y comparación
└── requirements.txt
```

---

## 4. Uso rápido

### Entrenar IQL en red 2×2 (el primer algoritmo a probar)
```bash
python train.py --algo iql --network 2x2 --demand moderate
```

### Ver la simulación en el GUI de SUMO (útil para depurar)
```bash
python train.py --algo iql --network 2x2 --gui --timesteps 5000
```

### Entrenar con 5 semillas distintas (para análisis estadístico)
```bash
python train.py --algo qmix --network 2x2 --demand moderate --runs 5
```

### Evaluar un modelo guardado
```bash
python evaluate.py --checkpoint logs/iql_2x2_moderate/run_1/best_model.pt
```

### Comparar todos los algoritmos
```bash
python evaluate.py --compare --network 2x2 --demand moderate --save_plots
```

---

## 5. Métricas evaluadas

| Métrica | Descripción |
|---------|-------------|
| **Tiempo de espera** | Segundos que cada vehículo espera en cola (↓ mejor) |
| **Longitud de cola** | Vehículos acumulados por carril (↓ mejor) |
| **Paradas/vehículo** | Cuántas veces frena cada vehículo (↓ mejor) |
| **Throughput** | Vehículos que cruzan la red por hora (↑ mejor) |
| **Índice de equidad** | Jain's Fairness Index entre intersecciones (↑ mejor) |

---

## 6. Orden de implementación recomendado

```
[✓] Paso 1 — Configuración y arranque      (este README)
[ ] Paso 2 — Entorno SUMO + Gymnasium      (env/)
[ ] Paso 3 — Algoritmos MARL               (agents/)
[ ] Paso 4 — Entrenamiento y buffer        (training/)
[ ] Paso 5 — Métricas y visualización      (evaluation/)
```

---

## 7. Solución de problemas frecuentes

**`ModuleNotFoundError: No module named 'traci'`**
```bash
pip install traci eclipse-sumo --force-reinstall
```

**`SUMO_HOME no definido`**
Ejecuta el paso 2.5 de esta guía.

**`RuntimeError: Cannot connect to SUMO`**
Asegúrate de que no tienes otra instancia de SUMO abierta. SUMO usa un puerto TCP local.

**Entrenamiento muy lento**
Usa `use_gui: false` en `config/default.yaml` (nunca entrenes con GUI activado).

---

## 8. Referencia de algoritmos

| Algoritmo | Tipo | Coordinación | Comunicación |
|-----------|------|-------------|--------------|
| **IQL** | Value-based | ✗ (independiente) | ✗ |
| **QMIX** | Value-based | ✓ (mixing net) | ✗ |
| **MAPPO** | Policy-based | ✓ (crítico central) | ✗ |
| **MADDPG** | Actor-Critic | ✓ (crítico central) | ✗ |
| **CommNet/TarMAC** | Value-based | ✓ | ✓ (mensajes) |
