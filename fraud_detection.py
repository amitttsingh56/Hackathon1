from __future__ import annotations
import time

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "transaction_id",
    "sender_id",
    "receiver_id",
    "amount",
    "timestamp",
]


def _to_dt_series(ts: pd.Series) -> pd.Series:
    # expected: YYYY-MM-DD HH:MM:SS
    return pd.to_datetime(ts, format="%Y-%m-%d %H:%M:%S", errors="coerce", utc=False)


def load_transactions(csv_bytes: bytes) -> pd.DataFrame:
    df = pd.read_csv(pd.io.common.BytesIO(csv_bytes))
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df[REQUIRED_COLUMNS].copy()
    out["timestamp"] = _to_dt_series(out["timestamp"])
    if out["timestamp"].isna().any():
        bad = out[out["timestamp"].isna()].head(5)["transaction_id"].tolist()
        raise ValueError(
            "Some timestamps could not be parsed as YYYY-MM-DD HH:MM:SS. "
            f"Example bad transaction_id(s): {bad}"
        )

    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    if out["amount"].isna().any():
        bad = out[out["amount"].isna()].head(5)["transaction_id"].tolist()
        raise ValueError(f"Some amounts are not numeric. Example transaction_id(s): {bad}")

    out["sender_id"] = out["sender_id"].astype(str)
    out["receiver_id"] = out["receiver_id"].astype(str)
    out["transaction_id"] = out["transaction_id"].astype(str)
    return out


def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    g = nx.DiGraph()
    # Store edge list per (u,v) to preserve multiple transfers.
    for row in df.itertuples(index=False):
        u = row.sender_id
        v = row.receiver_id
        attrs = {
            "transaction_id": row.transaction_id,
            "amount": float(row.amount),
            "timestamp": row.timestamp.to_pydatetime() if hasattr(row.timestamp, "to_pydatetime") else row.timestamp,
        }
        if g.has_edge(u, v):
            g[u][v]["transactions"].append(attrs)
            g[u][v]["total_amount"] += attrs["amount"]
            g[u][v]["count"] += 1
            if attrs["timestamp"] < g[u][v]["first_ts"]:
                g[u][v]["first_ts"] = attrs["timestamp"]
            if attrs["timestamp"] > g[u][v]["last_ts"]:
                g[u][v]["last_ts"] = attrs["timestamp"]
        else:
            g.add_edge(
                u,
                v,
                transactions=[attrs],
                total_amount=attrs["amount"],
                count=1,
                first_ts=attrs["timestamp"],
                last_ts=attrs["timestamp"],
            )
    return g


@dataclass(frozen=True)
class Finding:
    pattern: str
    ring_id: str
    nodes: Tuple[str, ...]
    edges: Tuple[Tuple[str, str], ...]
    score: float
    reasons: Tuple[str, ...]


def _node_degree_cache(g: nx.DiGraph) -> Dict[str, int]:
    deg = {}
    for n in g.nodes:
        deg[n] = int(g.in_degree(n) + g.out_degree(n))
    return deg


def _is_likely_merchant_or_payroll(
    node: str,
    deg: int,
    in_deg: int,
    out_deg: int,
    total_in: float,
    total_out: float,
    tx_in: int,
    tx_out: int,
) -> bool:
    """
    Heuristic suppression to reduce false positives:
    - very high degree and many counterparties
    - heavy, consistent inbound/outbound volumes (e.g., payroll / merchant aggregator)
    """
    # High-degree hubs (typical merchants / payroll processors / exchanges)
    if deg >= 80 and (in_deg >= 40 or out_deg >= 40):
        return True
    # Many transactions with roughly balanced flow suggests clearing account.
    if (tx_in + tx_out) >= 200:
        ratio = (total_in + 1.0) / (total_out + 1.0)
        if 0.7 <= ratio <= 1.3:
            return True
    return False


def _flow_stats(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    by_sender = df.groupby("sender_id").agg(
        tx_out=("transaction_id", "count"),
        total_out=("amount", "sum"),
        first_out=("timestamp", "min"),
        last_out=("timestamp", "max"),
        uniq_receivers=("receiver_id", "nunique"),
    )
    by_receiver = df.groupby("receiver_id").agg(
        tx_in=("transaction_id", "count"),
        total_in=("amount", "sum"),
        first_in=("timestamp", "min"),
        last_in=("timestamp", "max"),
        uniq_senders=("sender_id", "nunique"),
    )
    stats = by_sender.join(by_receiver, how="outer").fillna(
        {
            "tx_out": 0,
            "total_out": 0.0,
            "uniq_receivers": 0,
            "tx_in": 0,
            "total_in": 0.0,
            "uniq_senders": 0,
        }
    )
    stats.index = stats.index.astype(str)
    return stats


def detect_cycles_3_to_5(g: nx.DiGraph, max_cycles: int = 5000) -> List[Finding]:
    """
    Detect simple directed cycles with length 3-5.
    Performance notes:
    - `simple_cycles` can explode; we cap and early break.
    """
    findings: List[Finding] = []
    ring_idx = 0
    # Prune obvious high-degree hubs to avoid blow-ups + reduce false positives.
    # This keeps performance stable for ~10K transactions.
    deg = _node_degree_cache(g)
    # Increased pruning threshold from 60->500 to catch cycles in busier nodes
    pruned_nodes = [n for n, d in deg.items() if d <= 500 and g.in_degree(n) > 0 and g.out_degree(n) > 0]
    pg = g.subgraph(pruned_nodes).copy()

    for cyc in nx.simple_cycles(pg):
        k = len(cyc)
        if k < 3:
            continue
        if k > 5:
            continue

        # Build edges in order
        edges = []
        amounts = []
        for i in range(k):
            u = cyc[i]
            v = cyc[(i + 1) % k]
            edges.append((u, v))
            if g.has_edge(u, v):
                amounts.append(float(g[u][v]["total_amount"]))
        if not amounts:
            continue

        amt_cv = float(np.std(amounts) / (np.mean(amounts) + 1e-9))
        
        # Tightened CV threshold from 1.25->1.0 to reduce false positives with disparate amounts
        if amt_cv > 1.0:
            continue
            
        score = float(max(0.0, 1.0 - min(1.0, amt_cv)))  # similar amounts => higher
        reasons = (
            f"Directed cycle length={k}",
            f"Edge total amounts CV={amt_cv:.2f} (lower => more circular routing)",
        )
        findings.append(
            Finding(
                pattern="cycle_3_5",
                ring_id=f"CYCLE-{ring_idx:04d}",
                nodes=tuple(cyc),
                edges=tuple(edges),
                score=score,
                reasons=reasons,
            )
        )
        ring_idx += 1
        if len(findings) >= max_cycles:
            break
    return findings


def _windowed_partner_burst(
    df: pd.DataFrame,
    focal: str,
    mode: str,
    window_hours: int = 72,
    min_partners: int = 6,
    min_tx: int = 10,
    min_total_amount: float = 2000.0,
) -> Optional[Tuple[Set[str], pd.DataFrame, float, Tuple[str, ...]]]:
    """
    Smurfing detector:
    - fan-out: many distinct receivers in short time
    - fan-in: many distinct senders in short time
    """
    if mode == "fan_out":
        sub = df[df["sender_id"] == focal].sort_values("timestamp")
        partner_col = "receiver_id"
    else:
        sub = df[df["receiver_id"] == focal].sort_values("timestamp")
        partner_col = "sender_id"

    if sub.empty:
        return None

    ts = sub["timestamp"].to_numpy()
    partners = sub[partner_col].to_numpy()
    amounts = sub["amount"].to_numpy()

    j = 0
    best = None
    best_score = 0.0

    for i in range(len(sub)):
        while j < len(sub) and (ts[j] - ts[i]) <= np.timedelta64(window_hours, "h"):
            j += 1
        window = sub.iloc[i:j]
        uniq_partners = int(window[partner_col].nunique())
        tx_count = int(len(window))
        total_amt = float(window["amount"].sum())
        if uniq_partners < min_partners or tx_count < min_tx or total_amt < min_total_amount:
            continue

        # score: more partners + more tx => higher; penalize huge totals to avoid big merchants
        score = float(
            min(1.0, (uniq_partners / (min_partners * 2))) * 0.55
            + min(1.0, (tx_count / (min_tx * 2))) * 0.45
        )
        if score > best_score:
            best_score = score
            best = (set(window[partner_col].astype(str).unique().tolist()), window, total_amt)

    if not best:
        return None

    partners_set, window_df, total_amt = best
    reasons = (
        f"{mode} burst within {window_hours}h",
        f"unique_partners={len(partners_set)}, tx={len(window_df)}, total_amount={total_amt:.2f}",
    )
    return partners_set, window_df, best_score, reasons


def detect_smurfing(
    df: pd.DataFrame,
    g: nx.DiGraph,
    window_hours: int = 72,
    max_findings: int = 3000,
) -> List[Finding]:
    stats = _flow_stats(df)
    findings: List[Finding] = []
    ring_idx = 0

    # Candidate nodes: moderate degree, bursty counterparties
    deg_cache = _node_degree_cache(g)
    for node in g.nodes:
        deg = deg_cache.get(node, 0)
        in_deg = int(g.in_degree(node))
        out_deg = int(g.out_degree(node))
        s = stats.loc[node] if node in stats.index else None
        if s is None:
            continue

        if _is_likely_merchant_or_payroll(
            node=node,
            deg=deg,
            in_deg=in_deg,
            out_deg=out_deg,
            total_in=float(s.get("total_in", 0.0)),
            total_out=float(s.get("total_out", 0.0)),
            tx_in=int(s.get("tx_in", 0)),
            tx_out=int(s.get("tx_out", 0)),
        ):
            continue

        for mode in ("fan_out", "fan_in"):
            hit = _windowed_partner_burst(df, node, mode=mode, window_hours=window_hours)
            if not hit:
                continue
            partners_set, window_df, score, reasons = hit
            nodes = tuple([node] + sorted(partners_set))
            edges: List[Tuple[str, str]] = []
            if mode == "fan_out":
                edges = [(node, p) for p in partners_set]
            else:
                edges = [(p, node) for p in partners_set]

            findings.append(
                Finding(
                    pattern=f"smurfing_{mode}",
                    ring_id=f"SMURF-{ring_idx:04d}",
                    nodes=nodes,
                    edges=tuple(edges),
                    score=float(score),
                    reasons=tuple(reasons),
                )
            )
            ring_idx += 1
            if len(findings) >= max_findings:
                return findings
    return findings


def detect_layered_shell_networks(
    g: nx.DiGraph,
    max_findings: int = 4000,
    min_hops: int = 3,
    max_hops: int = 6,
    low_degree_max: int = 3,
) -> List[Finding]:
    """
    Layered shell networks:
    Look for directed paths with 3+ hops where intermediates have low total degree
    (throwaway accounts), connecting two higher-activity endpoints.

    Strategy:
    - Precompute degree.
    - For each low-degree node as potential intermediary, explore small DFS within hop bounds.
    - Cap results and prefer distinct paths.
    """
    deg = _node_degree_cache(g)
    low_nodes = [n for n, d in deg.items() if d <= low_degree_max]

    findings: List[Finding] = []
    ring_idx = 0

    # Build adjacency lists for speed
    succ = {n: list(g.successors(n)) for n in g.nodes}

    seen_paths: Set[Tuple[str, ...]] = set()

    def is_low_intermediate(x: str) -> bool:
        return deg.get(x, 0) <= low_degree_max

    for start in g.nodes:
        # Light pruning: only explore from nodes that send to something
        if g.out_degree(start) == 0:
            continue

        stack: List[Tuple[str, List[str]]] = [(start, [start])]
        while stack:
            cur, path = stack.pop()
            if len(path) - 1 >= max_hops:
                continue

            for nxt in succ.get(cur, []):
                if nxt in path:
                    continue
                new_path = path + [nxt]
                hops = len(new_path) - 1
                if hops >= min_hops:
                    # intermediates are path[1:-1]
                    inter = new_path[1:-1]
                    if inter and all(is_low_intermediate(x) for x in inter):
                        # endpoints should not be low-degree throwaways
                        if deg.get(new_path[0], 0) >= (low_degree_max + 2) or deg.get(new_path[-1], 0) >= (
                            low_degree_max + 2
                        ):
                            t = tuple(new_path)
                            if t in seen_paths:
                                continue
                            seen_paths.add(t)
                            edges = tuple((new_path[i], new_path[i + 1]) for i in range(len(new_path) - 1))
                            reasons = (
                                f"Directed chain hops={hops}",
                                f"All {len(inter)} intermediates have degree <= {low_degree_max}",
                            )
                            score = float(min(1.0, 0.25 + 0.15 * hops))
                            findings.append(
                                Finding(
                                    pattern="layered_shell_chain",
                                    ring_id=f"SHELL-{ring_idx:04d}",
                                    nodes=tuple(new_path),
                                    edges=edges,
                                    score=score,
                                    reasons=reasons,
                                )
                            )
                            ring_idx += 1
                            if len(findings) >= max_findings:
                                return findings

                stack.append((nxt, new_path))

    return findings


def consolidate_findings(
    findings: List[Finding],
    g: nx.DiGraph,
    top_k: int = 200,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, object]]]:
    """
    Produce:
    - a ring summary table
    - per-node suspicion summary for visualization
    """
    node_scores: Dict[str, float] = {n: 0.0 for n in g.nodes}
    node_reasons: Dict[str, List[str]] = {n: [] for n in g.nodes}

    for f in findings:
        for n in f.nodes:
            node_scores[n] = node_scores.get(n, 0.0) + float(f.score)
            node_reasons.setdefault(n, []).extend([f.pattern] + list(f.reasons))

    # Normalize for display
    scores = np.array(list(node_scores.values()), dtype=float)
    p95 = float(np.percentile(scores, 95)) if len(scores) else 1.0
    if p95 <= 0:
        p95 = 1.0

    node_summary: Dict[str, Dict[str, object]] = {}
    for n in g.nodes:
        s = float(node_scores.get(n, 0.0))
        node_summary[n] = {
            "suspicion_score": float(min(1.0, s / p95)),
            "raw_score": s,
            "reasons": list(dict.fromkeys(node_reasons.get(n, [])))[:20],
            "in_degree": int(g.in_degree(n)),
            "out_degree": int(g.out_degree(n)),
        }

    rows = []
    for f in findings:
        rows.append(
            {
                "ring_id": f.ring_id,
                "pattern": f.pattern,
                "score": f.score,
                "nodes": list(f.nodes),
                "num_nodes": len(f.nodes),
                "num_edges": len(f.edges),
                "reasons": list(f.reasons),
            }
        )
    rings_df = pd.DataFrame(rows).sort_values(["score", "num_nodes"], ascending=[False, False])
    if len(rings_df) > top_k:
        rings_df = rings_df.head(top_k).reset_index(drop=True)
    return rings_df, node_summary


def build_report_json(
    df: pd.DataFrame,
    findings: List[Finding],
    node_summary: Dict[str, Dict[str, object]],
    rings_df: pd.DataFrame,
    processing_time_seconds: float = 0.0,
) -> Dict[str, object]:

    # Map internal pattern names
    def map_pattern(pattern: str, nodes: Tuple[str, ...]) -> str:
        if pattern == "cycle_3_5":
            return f"cycle_length_{len(nodes)}"
        if pattern.startswith("smurfing"):
            return "high_velocity"
        if pattern == "layered_shell_chain":
            return "layered_shell"
        return pattern

    suspicious_accounts = []
    fraud_rings = []

    total_accounts = len(
        pd.unique(pd.concat([df["sender_id"], df["receiver_id"]]))
    )

    # Normalize suspicion scores
    max_raw = max([v["raw_score"] for v in node_summary.values()] + [1.0])

    # Track unique suspicious accounts to avoid duplicates across rings
    suspicious_map = {}

    for idx, f in enumerate(findings, start=1):

        clean_ring_id = f"RING_{idx:03d}"
        mapped_pattern = map_pattern(f.pattern, f.nodes)
        
        # Add ALL nodes involved in the finding, not just the best one
        for node in f.nodes:
            raw_score = node_summary.get(node, {}).get("raw_score", 0.0)
            suspicion_score = round((raw_score / max_raw) * 100, 1) if max_raw > 0 else 0.0
            
            if node not in suspicious_map:
                suspicious_map[node] = {
                    "account_id": node,
                    "suspicion_score": suspicion_score,
                    "detected_patterns": {mapped_pattern},
                    "ring_ids": {clean_ring_id}
                }
            else:
                # Merge patterns and rings
                suspicious_map[node]["detected_patterns"].add(mapped_pattern)
                suspicious_map[node]["ring_ids"].add(clean_ring_id)
                # Max score logic is already handled by node_summary aggregation, 
                # but we can ensure we display the high-level score.
                
        fraud_rings.append({
            "ring_id": clean_ring_id,
            "member_accounts": list(f.nodes),
            "pattern_type": mapped_pattern.split("_")[0],
            "risk_score": round(float(f.score) * 100, 1)
        })

    # Convert map to list and format sets as lists/strings
    suspicious_accounts = []
    for node, data in suspicious_map.items():
        data["detected_patterns"] = sorted(list(data["detected_patterns"]))
        # Join ring IDs or pick primary? Let's generic list them.
        # But to keep schema somewhat compatible (if it expects a string for ring_id), 
        # we can just take the first or comma join. 
        # The schema in README says "ring_id": "CYCLE-0000". 
        # I'll just use the first observed ring_id for backward compatibility, or comma string.
        data["ring_id"] = sorted(list(data["ring_ids"]))[0] 
        del data["ring_ids"]
        suspicious_accounts.append(data)

    # Sort by score descending
    suspicious_accounts.sort(key=lambda x: x["suspicion_score"], reverse=True)

    summary = {
        "total_accounts_analyzed": int(total_accounts),
        "suspicious_accounts_flagged": len(suspicious_accounts),
        "fraud_rings_detected": len(fraud_rings),
        "processing_time_seconds": round(processing_time_seconds, 1)
    }

    return {
        "suspicious_accounts": suspicious_accounts,
        "fraud_rings": fraud_rings,
        "summary": summary
    }