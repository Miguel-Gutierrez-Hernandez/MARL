"""
test_env.py — Prueba rápida del entorno
========================================
Ejecuta ESTO antes de cualquier entrenamiento para confirmar que SUMO
y el entorno funcionan correctamente.

Uso:
    python test_env.py
    python test_env.py --gui        # Con ventana gráfica de SUMO
    python test_env.py --steps 100  # Solo 100 pasos
"""

import argparse
import time

import numpy as np


def test_env(use_gui: bool = False, max_steps: int = 50):
    print("\n" + "═" * 55)
    print("  Test del entorno MultiAgentSumoEnv")
    print("═" * 55 + "\n")

    # 1. Verificar importaciones
    print("1. Importando módulos...")
    try:
        import sumo
        print(f"   ✓ SUMO {sumo.__version__}")
    except ImportError:
        print("   ✗ SUMO no encontrado. Ejecuta: pip install eclipse-sumo")
        return False

    try:
        import traci
        print(f"   ✓ traci importado")
    except ImportError:
        print("   ✗ traci no encontrado")
        return False

    try:
        from env.sumo_env import MultiAgentSumoEnv
        print(f"   ✓ MultiAgentSumoEnv importado")
    except ImportError as e:
        print(f"   ✗ Error importando el entorno: {e}")
        return False

    # 2. Verificar que existe la red
    print("\n2. Verificando archivos de red...")
    from pathlib import Path
    net_file = Path("networks/2x2/grid.net.xml")
    if not net_file.exists():
        print(f"   ✗ No existe: {net_file}")
        print("      Ejecuta primero: python networks/generate_networks.py")
        return False
    print(f"   ✓ {net_file} encontrado")

    # 3. Crear el entorno
    print("\n3. Creando entorno (red 2×2, demanda moderate)...")
    try:
        env = MultiAgentSumoEnv(
            network     = "2x2",
            demand      = "moderate",
            num_seconds = 300,       # Episodio corto para el test
            delta_time  = 5,
            use_gui     = use_gui,
        )
        print(f"   ✓ Entorno creado: {env}")
    except Exception as e:
        print(f"   ✗ Error creando entorno: {e}")
        return False

    # 4. Reset
    print("\n4. Ejecutando reset()...")
    try:
        t0 = time.time()
        obs, info = env.reset(seed=42)
        print(f"   ✓ Reset completado en {time.time()-t0:.2f}s")
        print(f"   ✓ Agentes: {env.agents}")
        print(f"   ✓ Num agentes: {env.num_agents}")
        for agent_id, o in obs.items():
            print(f"   ✓ obs[{agent_id}]: shape={o.shape}, "
                  f"min={o.min():.3f}, max={o.max():.3f}")
    except Exception as e:
        print(f"   ✗ Error en reset(): {e}")
        import traceback; traceback.print_exc()
        env.close()
        return False

    # 5. Loop de pasos
    print(f"\n5. Ejecutando {max_steps} pasos con acciones aleatorias...")
    total_reward = {a: 0.0 for a in env.agents}
    step = 0

    try:
        t0 = time.time()
        done = False

        while step < max_steps and not done:
            # Acciones aleatorias para cada agente
            actions = {
                agent_id: env.action_spaces[agent_id].sample()
                for agent_id in env.agents
            }

            obs, rewards, terminated, truncated, info = env.step(actions)

            for agent_id, r in rewards.items():
                total_reward[agent_id] += r

            done = all(terminated.values())
            step += 1

            if step % 10 == 0:
                global_info = info.get("__global__", {})
                wt = global_info.get("total_waiting_time", 0)
                q  = global_info.get("total_queue", 0)
                print(f"   Paso {step:3d} | espera total={wt:6.1f}s | cola={q:3.0f}")

        elapsed = time.time() - t0
        sps = step * env.delta_time / elapsed
        print(f"\n   ✓ {step} pasos completados en {elapsed:.2f}s "
              f"({sps:.0f} segundos simulados/s real)")

    except Exception as e:
        print(f"   ✗ Error durante step(): {e}")
        import traceback; traceback.print_exc()
        env.close()
        return False

    # 6. Estado global
    print("\n6. Verificando estado global (para CTDE)...")
    state = env.get_global_state()
    print(f"   ✓ Estado global: shape={state.shape}, "
          f"size esperado={env.global_state_size}")

    # 7. Close
    print("\n7. Cerrando entorno...")
    env.close()
    print("   ✓ SUMO cerrado correctamente")

    # Resumen
    print("\n" + "═" * 55)
    print("  ✓ TODOS LOS TESTS PASARON")
    print(f"  Recompensa acumulada por agente:")
    for agent_id, r in total_reward.items():
        print(f"    {agent_id}: {r:.4f}")
    print("═" * 55)
    print("\nEl entorno funciona. Puedes continuar con el Paso 3 (agentes MARL).\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui",   action="store_true", help="Abrir SUMO con interfaz gráfica")
    parser.add_argument("--steps", type=int, default=50, help="Número de pasos a ejecutar")
    args = parser.parse_args()

    success = test_env(use_gui=args.gui, max_steps=args.steps)
    exit(0 if success else 1)
