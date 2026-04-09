"""
Phase 2: Feature extraction for Vector Database.
Creates initial feature vectors h_i^(0) for actions:
  - One-hot encoding of activity label
  - Frequency in event log
  - Extended with Cost, HumanRes from node.csv and Role from human.csv
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class FeatureExtractor:
    """
    Extract feature vectors for events, human resources, and devices
    for storage in a Vector Database (Qdrant/Milvus).
    """

    def __init__(self):
        self.activity_encoder: Dict[str, int] = {}
        self.role_encoder: Dict[str, int] = {}
        self.device_type_encoder: Dict[str, int] = {}

    def fit(self, event_log_df: pd.DataFrame, node_data: Optional[list] = None):
        """Build encoders from event log and node metadata."""
        activities = sorted(event_log_df["concept:name"].unique())
        self.activity_encoder = {a: i for i, a in enumerate(activities)}

        if "org:resource" in event_log_df.columns:
            roles = sorted(event_log_df["org:resource"].dropna().unique())
            self.role_encoder = {r: i for i, r in enumerate(roles)}

    def extract_event_vectors(
        self, event_log_df: pd.DataFrame
    ) -> List[Dict]:
        """
        Create vector representations per unique (activity, graph) combination.

        Returns list of dicts with:
          - 'activity': str
          - 'graph_id': str
          - 'vector': np.ndarray
          - 'metadata': dict of additional info
        """
        num_activities = len(self.activity_encoder)
        freq = event_log_df.groupby(["concept:name", "graph_id"]).size().to_dict()
        total = len(event_log_df)

        results = []
        seen = set()

        for _, row in event_log_df.iterrows():
            key = (row["concept:name"], row["graph_id"])
            if key in seen:
                continue
            seen.add(key)

            act = row["concept:name"]
            gid = row["graph_id"]

            # One-hot + frequency
            vec = np.zeros(num_activities + 3, dtype=np.float32)
            if act in self.activity_encoder:
                vec[self.activity_encoder[act]] = 1.0
            vec[-3] = freq.get(key, 0) / total
            vec[-2] = row.get("cost", 0.0) / 500.0
            vec[-1] = 0.0  # placeholder for HumanRes

            results.append(
                {
                    "activity": act,
                    "graph_id": gid,
                    "vector": vec,
                    "metadata": {
                        "cost": row.get("cost", 0),
                        "resource": row.get("org:resource", ""),
                        "node_type": row.get("node_type", ""),
                    },
                }
            )

        return results

    def extract_resource_vectors(
        self, event_log_df: pd.DataFrame
    ) -> List[Dict]:
        """Create vector representations for human resources."""
        resources = event_log_df.groupby("org:resource").agg(
            activities=("concept:name", "nunique"),
            total_events=("concept:name", "count"),
            avg_cost=("cost", "mean"),
        ).reset_index()

        num_act = len(self.activity_encoder)
        results = []

        for _, row in resources.iterrows():
            name = row["org:resource"]
            if not name:
                continue
            vec = np.zeros(num_act + 3, dtype=np.float32)
            vec[-3] = row["total_events"] / len(event_log_df)
            vec[-2] = row["avg_cost"] / 500.0
            vec[-1] = row["activities"] / max(num_act, 1)

            results.append(
                {
                    "resource": name,
                    "vector": vec,
                    "metadata": {
                        "activities": int(row["activities"]),
                        "total_events": int(row["total_events"]),
                        "avg_cost": float(row["avg_cost"]),
                    },
                }
            )

        return results
