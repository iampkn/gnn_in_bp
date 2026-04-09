"""
Phase 1 Runner: Data Generation & Preprocessing
Reads base graphs from CSV, generates 20 variants, simulates event logs.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.phase1_data_generation import GraphReader, GraphGenerator, EventLogSimulator


def main():
    project_root = Path(__file__).resolve().parent.parent
    csv_dir = project_root / "src" / "data" / "csv"
    output_dir = project_root / "src" / "data" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PHASE 1: Data Generation & Preprocessing")
    print("=" * 60)

    # Step 1: Read base graphs
    print("\n[Step 1] Reading base graphs from CSV...")
    reader = GraphReader(csv_dir)
    base_graphs = reader.read_all()
    print(reader.summary(base_graphs))

    # Step 2: Generate 20 process variants
    print("\n[Step 2] Generating 20 process variants...")
    generator = GraphGenerator(seed=42)
    variant_graphs = generator.generate_variants(base_graphs, num_variants=20)
    print(f"Generated {len(variant_graphs)} graph variants: {list(variant_graphs.keys())}")

    all_graphs = {**base_graphs, **variant_graphs}
    print(f"Total graphs (base + variants): {len(all_graphs)}")

    # Step 3: Simulate event logs
    print("\n[Step 3] Simulating event logs (200 traces per graph)...")
    simulator = EventLogSimulator(seed=42)
    event_log_df = simulator.simulate(all_graphs, traces_per_graph=200)
    print(f"Total events generated: {len(event_log_df)}")
    print(f"Total cases (traces): {event_log_df['case:concept:name'].nunique()}")
    print(f"Unique activities: {event_log_df['concept:name'].nunique()}")

    # Step 4: Save outputs
    print("\n[Step 4] Saving event logs...")
    csv_output = output_dir / "event_log_all.csv"
    simulator.save_csv(event_log_df, str(csv_output))

    # Save per-graph event logs
    for gid in sorted(all_graphs.keys()):
        gid_df = event_log_df[event_log_df["graph_id"] == gid]
        if not gid_df.empty:
            gid_csv = output_dir / f"event_log_{gid}.csv"
            gid_df.to_csv(gid_csv, index=False, encoding="utf-8-sig")

    # Try XES export
    xes_output = output_dir / "event_log_all.xes"
    simulator.save_xes(event_log_df, str(xes_output))

    # Step 5: Summary statistics
    print("\n[Step 5] Summary Statistics")
    print("-" * 40)
    for gid in sorted(all_graphs.keys()):
        gid_df = event_log_df[event_log_df["graph_id"] == gid]
        n_cases = gid_df["case:concept:name"].nunique()
        n_events = len(gid_df)
        total_cost = gid_df["cost"].sum()
        print(f"  {gid}: {n_cases} cases, {n_events} events, total cost={total_cost:.0f}")

    print("\n" + "=" * 60)
    print("Phase 1 COMPLETE")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
