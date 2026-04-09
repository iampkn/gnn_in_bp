"""
Encode the process discovery problem as a DGL graph G.
G contains three parts (per the paper, Section III-B):
  1. Trace graph: event nodes from the event log
  2. Candidate Petri net: transitions + candidate places with arcs
  3. Links: edges from event nodes to transition nodes

Uses alpha-relations to prune candidate places (Appendix A of the paper).
"""
from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import dgl
import torch
import numpy as np
import pandas as pd


class AlphaRelations:
    """Compute alpha-relations from an event log for search space reduction."""

    def __init__(self, traces: List[List[str]], k_max: int = 3):
        self.traces = traces
        self.activities = sorted(set(a for tr in traces for a in tr))
        self.k_max = k_max
        self._directly_follows: Set[Tuple[str, str]] = set()
        self._eventually_follows: Dict[int, Set[Tuple[str, str]]] = {}
        self._compute_relations()

    def _compute_relations(self):
        for trace in self.traces:
            for i in range(len(trace) - 1):
                self._directly_follows.add((trace[i], trace[i + 1]))
                for k in range(1, min(self.k_max + 1, len(trace) - i)):
                    if k not in self._eventually_follows:
                        self._eventually_follows[k] = set()
                    self._eventually_follows[k].add((trace[i], trace[i + k]))

    def directly_follows(self, a: str, b: str) -> bool:
        return (a, b) in self._directly_follows

    def eventually_follows(self, a: str, b: str, k: Optional[int] = None) -> bool:
        if k is not None:
            return (a, b) in self._eventually_follows.get(k, set())
        return any(
            (a, b) in efs for efs in self._eventually_follows.values()
        )

    def causal(self, a: str, b: str) -> bool:
        return self.directly_follows(a, b) and not self.directly_follows(b, a)

    def parallel(self, a: str, b: str) -> bool:
        return self.directly_follows(a, b) and self.directly_follows(b, a)

    def conflict(self, a: str, b: str) -> bool:
        return not self.directly_follows(a, b) and not self.directly_follows(b, a)


class DiscoveryGraphEncoder:
    """
    Build the unified graph G for the GNN-based process discovery.

    Node types in G:
      - 'event': event nodes from traces
      - 'transition': one per unique activity
      - 'place': candidate places (X, Y) subsets of transitions

    Edge types:
      - event->event: trace sequence (bi-directional)
      - event->transition: links
      - transition->place, place->transition: Petri net arcs (bi-directional)
    """

    def __init__(self, k_max: int = 3, max_place_size: int = 2):
        self.k_max = k_max
        self.max_place_size = max_place_size

    def encode(
        self, event_log_df: pd.DataFrame, graph_id: Optional[str] = None
    ) -> dict:
        """
        Encode an event log DataFrame into the discovery graph structure.

        Returns dict with:
          - 'dgl_graph': the DGL heterogeneous graph
          - 'node_features': initial feature vectors
          - 'activity_to_idx': mapping of activity labels to indices
          - 'candidate_places': list of (input_set, output_set) tuples
          - 'transition_nodes': list of activity labels
          - 'metadata': additional info
        """
        if graph_id:
            df = event_log_df[event_log_df["graph_id"] == graph_id].copy()
        else:
            df = event_log_df.copy()

        traces = self._extract_traces(df)
        activities = sorted(set(a for tr in traces for a in tr))
        act_to_idx = {a: i for i, a in enumerate(activities)}
        num_activities = len(activities)

        alpha = AlphaRelations(traces, k_max=self.k_max)

        # Compute activity frequencies
        freq = defaultdict(int)
        for tr in traces:
            for a in tr:
                freq[a] += 1
        total_events = sum(freq.values())
        norm_freq = {a: freq[a] / total_events for a in activities}

        candidate_places = self._generate_candidate_places(
            activities, alpha
        )

        # Build node lists
        # Event nodes: one per event occurrence across all traces
        event_nodes = []
        for trace_idx, trace in enumerate(traces):
            for pos, act in enumerate(trace):
                event_nodes.append(
                    {"trace_idx": trace_idx, "pos": pos, "activity": act}
                )

        transition_nodes = activities
        place_nodes = candidate_places

        # Build initial feature vectors
        # Events: one-hot activity + normalized frequency
        event_features = []
        for ev in event_nodes:
            feat = np.zeros(num_activities + 1, dtype=np.float32)
            feat[act_to_idx[ev["activity"]]] = 1.0
            feat[-1] = norm_freq[ev["activity"]]
            event_features.append(feat)

        # Transitions: one-hot activity + frequency
        transition_features = []
        for act in transition_nodes:
            feat = np.zeros(num_activities + 1, dtype=np.float32)
            feat[act_to_idx[act]] = 1.0
            feat[-1] = norm_freq.get(act, 0.0)
            transition_features.append(feat)

        # Candidate places: zero vectors
        place_features = [
            np.zeros(num_activities + 1, dtype=np.float32)
            for _ in place_nodes
        ]

        # Build edges
        event_event_src, event_event_dst = [], []
        event_trans_src, event_trans_dst = [], []
        trans_place_src, trans_place_dst = [], []
        place_trans_src, place_trans_dst = [], []

        # Event->Event edges (within each trace, bi-directional)
        event_offset = 0
        for trace_idx, trace in enumerate(traces):
            trace_start = event_offset
            for pos in range(len(trace) - 1):
                src_idx = trace_start + pos
                dst_idx = trace_start + pos + 1
                event_event_src.extend([src_idx, dst_idx])
                event_event_dst.extend([dst_idx, src_idx])
            event_offset += len(trace)

        # Event->Transition links
        for ev_idx, ev in enumerate(event_nodes):
            trans_idx = act_to_idx[ev["activity"]]
            event_trans_src.append(ev_idx)
            event_trans_dst.append(trans_idx)

        # Transition<->Place edges
        for place_idx, (in_set, out_set) in enumerate(place_nodes):
            for a in in_set:
                t_idx = act_to_idx[a]
                trans_place_src.append(t_idx)
                trans_place_dst.append(place_idx)
                place_trans_src.append(place_idx)
                place_trans_dst.append(t_idx)
            for b in out_set:
                t_idx = act_to_idx[b]
                place_trans_src.append(place_idx)
                place_trans_dst.append(t_idx)
                trans_place_src.append(t_idx)
                trans_place_dst.append(place_idx)

        graph_data = {}
        if event_event_src:
            graph_data[("event", "seq", "event")] = (
                torch.tensor(event_event_src),
                torch.tensor(event_event_dst),
            )
        if event_trans_src:
            graph_data[("event", "link", "transition")] = (
                torch.tensor(event_trans_src),
                torch.tensor(event_trans_dst),
            )
        if trans_place_src:
            graph_data[("transition", "arc", "place")] = (
                torch.tensor(trans_place_src),
                torch.tensor(trans_place_dst),
            )
        if place_trans_src:
            graph_data[("place", "arc_rev", "transition")] = (
                torch.tensor(place_trans_src),
                torch.tensor(place_trans_dst),
            )

        if not graph_data:
            graph_data[("event", "seq", "event")] = (
                torch.tensor([0]),
                torch.tensor([0]),
            )

        g = dgl.heterograph(graph_data)

        feature_dim = num_activities + 1
        if len(event_features) > 0:
            g.nodes["event"].data["h"] = torch.tensor(
                np.array(event_features), dtype=torch.float32
            )
        if len(transition_features) > 0 and "transition" in g.ntypes:
            g.nodes["transition"].data["h"] = torch.tensor(
                np.array(transition_features), dtype=torch.float32
            )
        if len(place_features) > 0 and "place" in g.ntypes:
            g.nodes["place"].data["h"] = torch.tensor(
                np.array(place_features), dtype=torch.float32
            )

        return {
            "dgl_graph": g,
            "feature_dim": feature_dim,
            "activity_to_idx": act_to_idx,
            "candidate_places": candidate_places,
            "transition_nodes": transition_nodes,
            "num_events": len(event_nodes),
            "num_transitions": len(transition_nodes),
            "num_places": len(place_nodes),
            "traces": traces,
            "alpha_relations": alpha,
        }

    def _extract_traces(self, df: pd.DataFrame) -> List[List[str]]:
        traces = []
        for case_id, group in df.groupby("case:concept:name"):
            group_sorted = group.sort_values("time:timestamp")
            trace = group_sorted["concept:name"].tolist()
            traces.append(trace)
        return traces

    def _generate_candidate_places(
        self, activities: List[str], alpha: AlphaRelations
    ) -> List[Tuple[frozenset, frozenset]]:
        """
        Generate candidate places using alpha-relations (Appendix A).
        P^(1-1): one-to-one places based on k-eventually follows.
        P^(1-n), P^(n-1), P^(n-n): combined places without parallel pairs.
        """
        # P^(1-1): one-to-one places
        p_1_1 = set()
        for a in activities:
            for b in activities:
                if a != b and alpha.eventually_follows(a, b):
                    p_1_1.add((frozenset([a]), frozenset([b])))

        # P^(1-n): one-to-many
        p_1_n = set()
        for a in activities:
            outgoing = [b for b in activities if (frozenset([a]), frozenset([b])) in p_1_1]
            if len(outgoing) > 1:
                for size in range(2, min(len(outgoing) + 1, self.max_place_size + 1)):
                    for combo in itertools.combinations(outgoing, size):
                        if not any(
                            alpha.parallel(b1, b2)
                            for b1, b2 in itertools.combinations(combo, 2)
                        ):
                            p_1_n.add((frozenset([a]), frozenset(combo)))

        # P^(n-1): many-to-one
        p_n_1 = set()
        for b in activities:
            incoming = [a for a in activities if (frozenset([a]), frozenset([b])) in p_1_1]
            if len(incoming) > 1:
                for size in range(2, min(len(incoming) + 1, self.max_place_size + 1)):
                    for combo in itertools.combinations(incoming, size):
                        if not any(
                            alpha.parallel(a1, a2)
                            for a1, a2 in itertools.combinations(combo, 2)
                        ):
                            p_n_1.add((frozenset(combo), frozenset([b])))

        all_places = list(p_1_1 | p_1_n | p_n_1)
        return all_places
