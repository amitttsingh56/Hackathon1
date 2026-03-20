from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
from pyvis.network import Network


def _color_for_score(score: float) -> str:
    # score in [0,1]
    score = max(0.0, min(1.0, float(score)))
    # green -> yellow -> red
    if score < 0.4:
        return "#5CB85C"
    if score < 0.7:
        return "#F0AD4E"
    return "#D9534F"


def build_pyvis_html(
    g: nx.DiGraph,
    node_summary: Dict[str, Dict[str, object]],
    highlight_nodes: Optional[Iterable[str]] = None,
    highlight_edges: Optional[Iterable[Tuple[str, str]]] = None,
    height_px: int = 650,
) -> str:
    highlight_nodes_set = set(highlight_nodes or [])
    highlight_edges_set = set(highlight_edges or [])

    net = Network(height=f"{height_px}px", width="100%", directed=True, bgcolor="#0E1117", font_color="#E6E6E6")
    net.barnes_hut(gravity=-20000, central_gravity=0.3, spring_length=150, spring_strength=0.06, damping=0.35)

    # Nodes
    for n in g.nodes:
        s = node_summary.get(n, {})
        suspicion = float(s.get("suspicion_score", 0.0))
        color = _color_for_score(suspicion)
        if n in highlight_nodes_set:
            border = "#FFFFFF"
            border_width = 3
        else:
            border = "#333333"
            border_width = 1
        title = {
            "node_id": n,
            "suspicion_score": suspicion,
            "raw_score": float(s.get("raw_score", 0.0)),
            "in_degree": int(s.get("in_degree", 0)),
            "out_degree": int(s.get("out_degree", 0)),
            "reasons": s.get("reasons", [])[:10],
        }
        net.add_node(
            n,
            label=n,
            title=f"<pre style='white-space:pre-wrap'>{json.dumps(title, indent=2)}</pre>",
            color={"background": color, "border": border},
            borderWidth=border_width,
            size=10 + 18 * suspicion,
        )

    # Edges
    for u, v, data in g.edges(data=True):
        is_hl = (u, v) in highlight_edges_set
        width = 1.0 + (1.5 if is_hl else 0.0)
        color = "#77B7FF" if is_hl else "rgba(160,160,160,0.35)"
        title = {
            "from": u,
            "to": v,
            "count": int(data.get("count", 1)),
            "total_amount": float(data.get("total_amount", 0.0)),
            "first_ts": str(data.get("first_ts", "")),
            "last_ts": str(data.get("last_ts", "")),
        }
        net.add_edge(u, v, value=float(data.get("total_amount", 0.0)), width=width, color=color, title=json.dumps(title))

    net.set_options(
        """
        var options = {
          "nodes": {"shape": "dot"},
          "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.6}},
            "smooth": {"type": "dynamic"}
          },
          "interaction": {"hover": true, "multiselect": true, "navigationButtons": true},
          "physics": {"enabled": true}
        }
        """
    )

    return net.generate_html()

