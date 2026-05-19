"""
networks/generate_networks.py
=============================
Genera las redes viarias 2×2, 3×3 y 4×4 usando la herramienta `netgenerate`
que viene incluida en SUMO.

Uso:
    python networks/generate_networks.py

Esto crea los archivos .net.xml en cada subcarpeta.
Después, demand.py generará los .rou.xml para cada nivel de tráfico.

¿Qué es un archivo .net.xml?
    La definición de la red viaria: nodos (intersecciones), aristas (calles),
    carriles, semáforos y su lógica de fases.
"""

import os
import subprocess
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

# ── Configuración de las redes ────────────────────────────────────────────────
NETWORKS = {
    "2x2": {"number": 2, "agents": 4},   # 4 intersecciones
    "3x3": {"number": 3, "agents": 9},   # 9 intersecciones
    "4x4": {"number": 4, "agents": 16},  # 16 intersecciones
}

ROOT = Path(__file__).parent.parent
NETWORKS_DIR = Path(__file__).parent

def find_executable(name: str) -> str:
    """Localiza un ejecutable incluido en eclipse-sumo."""
    import sumo
    sumo_home = Path(sumo.__file__).parent
    candidates = [
        sumo_home / "bin" / name,
        sumo_home / "bin" / f"{name}.exe",      # Windows
        Path(f"/usr/bin/{name}"),               # Linux
        Path(f"/usr/local/bin/{name}"),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return name

def generate_grid(name: str, number: int) -> None:
    out_dir = NETWORKS_DIR / name
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "grid.net.xml"
    plain_prefix = out_dir / "plain"

    netgenerate = find_executable("netgenerate")
    netconvert = find_executable("netconvert")

    # 1. Generar la cuadrícula en archivos XML planos
    cmd_gen = [
        netgenerate,
        "--grid",
        f"--grid.number={number}",
        "--grid.length=300",
        "--default.lanenumber=2",
        "--default.speed=13.89",
        "--no-turnarounds=true",
        "--junctions.corner-detail=5",
        f"--plain-output-prefix={plain_prefix}",
        "--no-warnings=true",
    ]

    print(f"  Generando estructura base {name} ({number}×{number}) en modo texto...")
    result = subprocess.run(cmd_gen, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] netgenerate falló:\n{result.stderr}")
        sys.exit(1)

    # 2. Modificar el archivo de nodos puros con Python (¡Estrategia universal!)
    nod_file = out_dir / "plain.nod.xml"
    tree = ET.parse(nod_file)
    root = tree.getroot()
    
    tls_count = 0
    for node in root.findall("node"):
        # Cualquier nodo que sea una intersección (que no sea un final de calle 'dead_end')
        # lo convertimos a la fuerza en semáforo.
        if node.get("type", "") != "dead_end":
            node.set("type", "traffic_light")
            tls_count += 1
            
    # Guardamos el XML modificado
    tree.write(nod_file, encoding="utf-8", xml_declaration=True)
    print(f"  > Se han forzado {tls_count} nodos a tipo 'traffic_light'")

    # 3. Compilar los archivos planos ya modificados para crear el mapa real
    cmd_conv = [
        netconvert,
        f"--node-files={nod_file}",
        f"--edge-files={out_dir / 'plain.edg.xml'}",
        f"--output-file={out_file}",
        "--no-warnings=true",
    ]
    
    con_file = out_dir / "plain.con.xml"
    if con_file.exists():
        cmd_conv.append(f"--connection-files={con_file}")

    print(f"  Ensamblando la red final...")
    result_conv = subprocess.run(cmd_conv, capture_output=True, text=True)
    
    if result_conv.returncode != 0:
        print(f"  [ERROR] netconvert falló al ensamblar:\n{result_conv.stderr}")
        sys.exit(1)

    # 4. Limpieza de archivos temporales
    for ext in ["nod.xml", "edg.xml", "con.xml", "tll.xml", "typ.xml"]:
        f = out_dir / f"plain.{ext}"
        if f.exists():
            f.unlink()

    print(f"  ✓ Red creada y ensamblada con éxito: {out_file}")

def write_sumocfg(name: str) -> None:
    out_dir = NETWORKS_DIR / name
    cfg_path = out_dir / "grid.sumocfg"
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <input>
        <net-file value="grid.net.xml"/>
        <route-files value="grid.rou.xml"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="1"/>
    </time>
    <processing>
        <ignore-route-errors value="true"/>
        <collision.action value="warn"/>
    </processing>
    <report>
        <no-step-log value="true"/>
        <no-warnings value="true"/>
    </report>
</configuration>
"""
    cfg_path.write_text(content, encoding="utf-8")
    print(f"  ✓ Configuración creada: {cfg_path}")

def main():
    print("\n" + "═" * 55)
    print("  Generador de redes viarias SUMO — marl_traffic")
    print("═" * 55 + "\n")

    try:
        import sumo
        print(f"SUMO detectado: versión {sumo.__version__}\n")
    except ImportError:
        print("[ERROR] SUMO no está instalado.\nEjecuta: pip install eclipse-sumo")
        sys.exit(1)

    for name, params in NETWORKS.items():
        print(f"── Red {name} ──────────────────────────────────────")
        generate_grid(name, params["number"])
        write_sumocfg(name)
        print()

    print("✓ Todas las redes generadas y procesadas con éxito.\n")

if __name__ == "__main__":
    main()