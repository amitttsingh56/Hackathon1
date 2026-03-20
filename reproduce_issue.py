
import pandas as pd
import networkx as nx
from fraud_detection import build_graph, detect_cycles_3_to_5

def run_test():
    print("Loading test transactions...")
    with open("test_transactions.csv", "rb") as f:
        df = pd.read_csv(f)
    
    # Simple preprocessing as in load_transactions
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    print(f"Loaded {len(df)} transactions.")
    
    g = build_graph(df)
    print(f"Graph stats: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges.")
    
    # Add dummy high-degree connections to node P to test pruning
    # P is part of a valid cycle P->Q->R->P
    # We add 70 incoming edges to P to make its degree > 60
    print("Adding high degree noise to node P...")
    for i in range(70):
        dummy_node = f"Z_IN_{i}"
        g.add_edge(dummy_node, "P", transactions=[], total_amount=10.0, count=1, first_ts=pd.Timestamp("2023-01-01"), last_ts=pd.Timestamp("2023-01-01"))
        
    print(f"Node P degree: {g.degree('P')}")
    
    print("Detecting cycles...")
    findings = detect_cycles_3_to_5(g)
    
    print(f"Found {len(findings)} cycle findings.")
    for f in findings:
        print(f" - {f.pattern}: {f.nodes} (Score: {f.score:.2f})")
        
    expected_cycles = {
        tuple(sorted(["A", "B", "C"])),
        tuple(sorted(["D", "E", "F", "G"])),
        tuple(sorted(["H", "I", "J", "K", "L"])),
        tuple(sorted(["P", "Q", "R"]))
    }
    
    found_cycles = set()
    for f in findings:
        found_cycles.add(tuple(sorted(f.nodes)))
        
    print("\nVerification:")
    for nodes in expected_cycles:
        if nodes in found_cycles:
            print(f" [PASS] Cycle {nodes} found.")
        else:
            print(f" [FAIL] Cycle {nodes} NOT found.")

    # Check for the filtered one
    bad_cycle = tuple(sorted(["M", "N", "O"]))
    if bad_cycle in found_cycles:
        print(f" [FAIL] Cycle {bad_cycle} (high variance) was found but should be filtered.")
    else:
        print(f" [PASS] Cycle {bad_cycle} (high variance) was correctly filtered.")

    # Verify suspicious account count
    from fraud_detection import consolidate_findings, build_report_json
    rings_df, node_summary = consolidate_findings(findings, g)
    report = build_report_json(df, findings, node_summary, rings_df)
    
    suspicious_count = len(report["suspicious_accounts"])
    expected_count = 3+4+5+3 # A-C, D-G, H-L, P-R
    print(f"\nSuspicious Accounts Count: {suspicious_count} (Expected: {expected_count})")
    
    if suspicious_count == expected_count:
        print(" [PASS] Suspicious account count is correct (includes all cycle members).")
    else:
        print(" [FAIL] Suspicious account count is incorrect.")

if __name__ == "__main__":
    run_test()
