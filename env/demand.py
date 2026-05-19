"""
env/demand.py
=============
Genera el archivo de rutas (.rou.xml) que SUMO necesita para saber
cuántos vehículos circulan, desde dónde y adónde van.

Conceptos SUMO:
  - <vType>   : tipo de vehículo (coche, camión…) con sus parámetros físicos
  - <flow>    : flujo continuo de vehículos en una ruta durante un intervalo
  - <route>   : secuencia de calles (edges) desde origen a destino

La distribución de llegadas sigue un proceso de Poisson, lo que es estándar
en modelos de tráfico. El parámetro `probability` de SUMO indica la
probabilidad de que aparezca un vehículo en cada segundo de simulación.
"""

import itertools
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional


# ── Tasas de tráfico por nivel de demanda ─────────────────────────────────────
# probability = P(un vehículo aparece en ese flujo en cada segundo)
# ~3600 * probability = vehículos por hora por flujo

DEMAND_LEVELS = {
    "low":       0.05,    # ≈ 180 veh/h por flujo de entrada → tráfico fluido
    "moderate":  0.12,    # ≈ 430 veh/h por flujo             → tráfico normal
    "saturated": 0.22,    # ≈ 790 veh/h por flujo             → colas habituales
}


def get_border_edges(net_xml_path: str) -> tuple[list[str], list[str]]:
    """
    Lee el .net.xml y extrae las aristas (edges) en el borde de la red.
    Estas son las que los vehículos usan como origen y destino.

    Una arista "borde" tiene solo un extremo conectado a la red principal.
    Las reconocemos porque su nodo origen o destino empieza con "-" (borde externo).
    """
    tree = ET.parse(net_xml_path)
    root = tree.getroot()

    sources = []   # Edges donde nacen vehículos (entran a la red)
    sinks   = []   # Edges donde mueren vehículos (salen de la red)

    for edge in root.findall("edge"):
        edge_id = edge.get("id", "")
        # SUMO genera aristas internas con ":" y conectoras con "#"
        if edge_id.startswith(":") or "#" in edge_id:
            continue

        from_node = edge.get("from", "")
        to_node   = edge.get("to", "")

        # Los nodos de borde en netgenerate tienen coordenadas extremas
        # Una heurística simple: si la función de la arista es "normal" (sin función)
        # y es una arista corta de borde la detectamos por los nodos
        func = edge.get("function", "")
        if func in ("internal", "connector"):
            continue

        # En redes generadas con netgenerate, las aristas de borde
        # conectan nodos externos ("-gneN") con nodos interiores
        if from_node.startswith("-") or from_node.startswith("gne"):
            sources.append(edge_id)
        if to_node.startswith("-") or to_node.startswith("gne"):
            sinks.append(edge_id)

    # Fallback: si la heurística no funcionó bien, usar todas las aristas
    # no internas como posibles orígenes/destinos
    if not sources:
        all_edges = [
            e.get("id") for e in root.findall("edge")
            if not e.get("id", "").startswith(":")
            and e.get("function", "") not in ("internal", "connector")
        ]
        sources = all_edges[:len(all_edges)//2]
        sinks   = all_edges[len(all_edges)//2:]

    return sources, sinks


def generate_demand(
    net_xml_path: str,
    output_rou_path: str,
    demand_level: str = "moderate",
    num_seconds: int = 3600,
    seed: Optional[int] = 42,
) -> str:
    """
    Genera el archivo .rou.xml con los flujos de vehículos.

    Parámetros
    ----------
    net_xml_path    : ruta al .net.xml de la red
    output_rou_path : ruta de salida para el .rou.xml
    demand_level    : "low" | "moderate" | "saturated"
    num_seconds     : duración del episodio en segundos simulados
    seed            : semilla aleatoria para reproducibilidad

    Devuelve
    --------
    La ruta al archivo .rou.xml generado.
    """
    if seed is not None:
        random.seed(seed)

    prob = DEMAND_LEVELS.get(demand_level, DEMAND_LEVELS["moderate"])
    sources, sinks = get_border_edges(net_xml_path)

    # Construir el XML
    root = ET.Element("routes")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation",
             "http://sumo.dlr.de/xsd/routes_file.xsd")

    # ── Tipo de vehículo estándar ─────────────────────────────────────────────
    vtype = ET.SubElement(root, "vType")
    vtype.set("id", "car")
    vtype.set("accel", "2.6")          # Aceleración máxima (m/s²)
    vtype.set("decel", "4.5")          # Deceleración máxima (m/s²)
    vtype.set("sigma", "0.5")          # Variabilidad del conductor (0=robot, 1=humano)
    vtype.set("length", "5.0")         # Longitud del vehículo (m)
    vtype.set("maxSpeed", "13.89")     # 50 km/h en m/s
    vtype.set("guiShape", "passenger")

    # ── Generar flujos origen→destino ────────────────────────────────────────
    # Creamos un flujo por cada par (origen, destino) distinto
    flow_id = 0
    for src, dst in itertools.product(sources, sinks):
        if src == dst:
            continue  # No tiene sentido origen = destino

        # Variar ligeramente la probabilidad para heterogeneidad
        p = max(0.01, prob + random.uniform(-0.02, 0.02))

        flow = ET.SubElement(root, "flow")
        flow.set("id",          f"flow_{flow_id}")
        flow.set("type",        "car")
        flow.set("from",        src)
        flow.set("to",          dst)
        flow.set("begin",       "0")
        flow.set("end",         str(num_seconds))
        flow.set("probability", f"{p:.4f}")   # Proceso de Poisson
        flow.set("departLane",  "random")     # Carril de salida aleatorio
        flow.set("departSpeed", "random")     # Velocidad inicial aleatoria
        flow_id += 1

    # ── Escribir el archivo ───────────────────────────────────────────────────
    output_path = Path(output_rou_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")          # Formatear con indentación (Python 3.9+)
    tree.write(str(output_path), encoding="unicode", xml_declaration=True)

    n_flows = flow_id
    print(f"  [demand] {demand_level} → {n_flows} flujos O/D, "
          f"prob≈{prob:.2f} ({prob*3600:.0f} veh/h por flujo)")

    return str(output_path)


# ── Test rápido ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    net = sys.argv[1] if len(sys.argv) > 1 else "networks/2x2/grid.net.xml"
    out = sys.argv[2] if len(sys.argv) > 2 else "networks/2x2/grid.rou.xml"
    level = sys.argv[3] if len(sys.argv) > 3 else "moderate"

    result = generate_demand(net, out, demand_level=level)
    print(f"✓ Rutas generadas: {result}")
