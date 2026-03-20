from __future__ import annotations

import os
import sys
import json
import time

# Support local .deps folder (optional)
_DEPS_DIR = os.path.join(os.path.dirname(__file__), ".deps")
if os.path.isdir(_DEPS_DIR) and _DEPS_DIR not in sys.path:
    sys.path.insert(0, _DEPS_DIR)

import pandas as pd
import streamlit as st

from fraud_detection import (
    build_graph,
    build_report_json,
    consolidate_findings,
    detect_cycles_3_to_5,
    detect_layered_shell_networks,
    detect_smurfing,
    load_transactions,
)
from graph_viz import build_pyvis_html


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Fraud Intelligence Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM DARK FINTECH UI
# --------------------------------------------------
st.markdown("""
<style>

.stApp {
    background-color: #0f172a;
}

h1, h2, h3, h4 {
    color: white !important;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    padding: 20px;
    border-radius: 16px;
    text-align: center;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
    color: white;
}

.metric-card h4 {
    margin-bottom: 5px;
    font-weight: 400;
}

.metric-card h2 {
    margin-top: 5px;
    font-weight: 700;
}

.stDownloadButton button {
    background-color: #6366f1;
    color: white;
    border-radius: 8px;
    height: 42px;
}

[data-testid="stDataFrame"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DOWNLOAD BUTTON
# --------------------------------------------------
def download_json_button(data: dict, filename: str):
    b = json.dumps(data, indent=2).encode("utf-8")
    st.download_button(
        label="Download JSON Report",
        data=b,
        file_name=filename,
        mime="application/json",
        use_container_width=True,
    )


# --------------------------------------------------
# MAIN APP
# --------------------------------------------------
def main():

    st.markdown("Money Muling Intelligence Dashboard")
    st.caption("Graph-Based Financial Crime Detection • Cycles • Smurfing • Shell Chains")

    # ---------------- Sidebar ----------------
    with st.sidebar:

        st.header("Upload Data")
        file = st.file_uploader("Transactions CSV", type=["csv"])

        st.header("Detection Settings")

        window_hours = st.slider("Smurfing Window (Hours)", 12, 168, 72, step=12)
        max_cycles = st.slider("Max Cycles (3–5)", 100, 20000, 5000, step=100)
        max_smurf = st.slider("Max Smurfing Findings", 100, 20000, 3000, step=100)
        max_shell = st.slider("Max Shell Chains", 100, 20000, 4000, step=100)
        low_degree_max = st.slider("Shell Max Degree", 1, 10, 3)
        max_hops = st.slider("Shell Max Hops", 4, 10, 6)
        top_k_rings = st.slider("Top Ring Rows", 20, 500, 200, step=20)

        st.header("Graph View")
        graph_height = st.slider("Graph Height (px)", 450, 1100, 650, step=50)
        min_suspicion_to_highlight = st.slider(
            "Highlight Suspicion ≥", 0.0, 1.0, 0.55, step=0.05
        )

    if not file:
        st.info("Upload a CSV file to begin analysis.")
        return

    # ---------------- Load Data ----------------
    try:
        t0 = time.time()
        df = load_transactions(file.getvalue())
        st.success(f"Loaded {len(df):,} transactions successfully.")
    except Exception as e:
        st.error(str(e))
        return

    # ---------------- Build Graph ----------------
    with st.spinner("Building transaction graph..."):
        g = build_graph(df)

    # ---------------- Metric Cards ----------------
    col1, col2, col3, col4 = st.columns(4)

    col1.markdown(f"""
    <div class="metric-card">
        <h4>Nodes</h4>
        <h2>{g.number_of_nodes():,}</h2>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="metric-card">
        <h4>Edges</h4>
        <h2>{g.number_of_edges():,}</h2>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="metric-card">
        <h4>Transactions</h4>
        <h2>{len(df):,}</h2>
    </div>
    """, unsafe_allow_html=True)

    col4.markdown(f"""
    <div class="metric-card">
        <h4>Load Time</h4>
        <h2>{(time.time() - t0):.2f}s</h2>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ---------------- Detection ----------------
    with st.spinner("Detecting fraud patterns..."):

        findings = []
        findings.extend(detect_cycles_3_to_5(g, max_cycles=int(max_cycles)))
        findings.extend(detect_smurfing(df, g, window_hours=int(window_hours), max_findings=int(max_smurf)))
        findings.extend(
            detect_layered_shell_networks(
                g,
                max_findings=int(max_shell),
                min_hops=3,
                max_hops=int(max_hops),
                low_degree_max=int(low_degree_max),
            )
        )

        rings_df, node_summary = consolidate_findings(findings, g, top_k=int(top_k_rings))

        report = build_report_json(df, findings, node_summary, rings_df)

    num_cycles = sum(1 for f in findings if f.pattern == "cycle_3_5")
    num_suspicious = len(report.get("suspicious_accounts", []))
    st.success(f"Detection complete. Found {num_cycles:,} cycles and {num_suspicious:,} suspicious accounts.")

    # ---------------- Highlight Logic ----------------
    # Highlight ANY node involved in a finding
    finding_nodes = set()
    for f in findings:
        finding_nodes.update(f.nodes)
        
    highlight_nodes = list(finding_nodes)

    highlight_edges = set()
    for f in findings:
        for u, v in f.edges:
            highlight_edges.add((u, v))

    # ---------------- Layout ----------------
    left, right = st.columns([0.45, 0.55], gap="large")

    with left:
        st.subheader("Fraud Ring Summary")

        if rings_df.empty:
            st.warning("No suspicious rings detected.")
        else:
            st.dataframe(rings_df, use_container_width=True, height=520)

        download_json_button(report, "fraud_report.json")

    with right:
        st.subheader("Interactive Network Graph")

        html = build_pyvis_html(
            g,
            node_summary,
            highlight_nodes=highlight_nodes,
            highlight_edges=highlight_edges,
            height_px=int(graph_height),
        )

        st.components.v1.html(html, height=int(graph_height) + 40, scrolling=True)

    st.caption("© 2026 ReinforceX. All Rights Reserved.")


if __name__ == "__main__":
    main()
