"""
Simulate event logs from ProcessGraph objects.
Handles ExclusiveGateway branching to produce diverse traces.
Outputs pm4py-compatible DataFrames and XES files.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .graph_reader import ProcessGraph


class EventLogSimulator:
    """
    Walk each process graph from StartEvent to EndEvent, handling
    ExclusiveGateway nodes by randomly choosing an outgoing branch.
    Produces an event log as a pandas DataFrame with columns:
      case:concept:name, concept:name, time:timestamp, org:resource,
      cost, graph_id
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def simulate(
        self,
        graphs: Dict[str, ProcessGraph],
        traces_per_graph: int = 200,
        base_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Generate event logs for all provided graphs."""
        if base_time is None:
            base_time = datetime(2026, 1, 1, 8, 0, 0)

        all_events: List[dict] = []
        case_counter = 0

        for gid, graph in sorted(graphs.items()):
            for _ in range(traces_per_graph):
                case_counter += 1
                case_id = f"case_{case_counter}"
                trace_events = self._simulate_single_trace(
                    graph, case_id, base_time, gid
                )
                all_events.extend(trace_events)
                base_time += timedelta(minutes=self.rng.randint(5, 60))

        df = pd.DataFrame(all_events)
        if not df.empty:
            df["time:timestamp"] = pd.to_datetime(df["time:timestamp"])
            df = df.sort_values(
                ["case:concept:name", "time:timestamp"]
            ).reset_index(drop=True)
        return df

    def _simulate_single_trace(
        self,
        graph: ProcessGraph,
        case_id: str,
        base_time: datetime,
        graph_id: str,
        max_steps: int = 50,
    ) -> List[dict]:
        events = []
        current_node = graph.start_node
        if current_node is None:
            return events

        timestamp = base_time + timedelta(minutes=self.rng.randint(0, 120))
        visited_count: Dict[str, int] = {}
        step = 0

        while current_node is not None and step < max_steps:
            step += 1
            node_info = graph.nodes.get(current_node)
            if node_info is None:
                break

            visited_count[current_node] = visited_count.get(current_node, 0) + 1

            if node_info.node_type != "ExclusiveGateway":
                resource = ""
                if node_info.humans:
                    resource = node_info.humans[0].get("name", "")

                events.append(
                    {
                        "case:concept:name": case_id,
                        "concept:name": node_info.label,
                        "time:timestamp": timestamp.isoformat(),
                        "org:resource": resource,
                        "cost": node_info.cost,
                        "graph_id": graph_id,
                        "node_type": node_info.node_type,
                    }
                )

                duration = self._estimate_duration(node_info.cost)
                timestamp += timedelta(minutes=duration)

            successors = graph.get_successors(current_node)
            if not successors:
                break

            if node_info.node_type == "ExclusiveGateway":
                current_node = self._choose_gateway_branch(
                    graph, current_node, successors, visited_count
                )
            else:
                current_node = successors[0]

            if node_info.node_type == "EndEvent":
                break

        return events

    def _choose_gateway_branch(
        self,
        graph: ProcessGraph,
        gateway_id: str,
        successors: List[str],
        visited_count: Dict[str, int],
    ) -> str:
        """
        At an ExclusiveGateway, choose one outgoing branch.
        Bias towards forward progress if a loop target has been visited many times.
        """
        loop_targets = []
        forward_targets = []

        for succ in successors:
            if visited_count.get(succ, 0) > 0:
                loop_targets.append(succ)
            else:
                forward_targets.append(succ)

        if not forward_targets:
            return self.rng.choice(successors)

        if not loop_targets:
            return self.rng.choice(successors)

        max_visits = max(visited_count.get(lt, 0) for lt in loop_targets)
        # Probability of looping decreases as visit count increases
        loop_prob = max(0.1, 0.5 ** max_visits)

        if self.rng.random() < loop_prob:
            return self.rng.choice(loop_targets)
        else:
            return self.rng.choice(forward_targets)

    def _estimate_duration(self, cost: float) -> int:
        """Estimate activity duration in minutes based on cost."""
        if cost <= 0:
            return 1
        base_minutes = max(5, int(cost / 10))
        jitter = self.rng.randint(-base_minutes // 3, base_minutes // 3)
        return max(1, base_minutes + jitter)

    @staticmethod
    def to_pm4py_log(df: pd.DataFrame):
        """Convert DataFrame to pm4py EventLog object."""
        try:
            import pm4py
            from pm4py.objects.conversion.log import converter as log_converter
            from pm4py.objects.log.util import dataframe_utils

            df = dataframe_utils.convert_timestamp_columns_in_df(df)
            params = {
                log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "case:concept:name"
            }
            event_log = log_converter.apply(df, parameters=params)
            return event_log
        except ImportError:
            print("pm4py not installed. Returning DataFrame instead.")
            return df

    @staticmethod
    def save_xes(df: pd.DataFrame, filepath: str):
        """Save event log as XES file using pm4py."""
        try:
            import pm4py
            from pm4py.objects.conversion.log import converter as log_converter
            from pm4py.objects.log.util import dataframe_utils

            df_copy = df.copy()
            df_copy = dataframe_utils.convert_timestamp_columns_in_df(df_copy)
            params = {
                log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "case:concept:name"
            }
            event_log = log_converter.apply(df_copy, parameters=params)
            pm4py.write_xes(event_log, filepath)
            print(f"Saved XES to {filepath}")
        except ImportError:
            csv_path = filepath.replace(".xes", ".csv")
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"pm4py not available. Saved CSV to {csv_path}")

    @staticmethod
    def save_csv(df: pd.DataFrame, filepath: str):
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
        print(f"Saved CSV event log to {filepath}")
