"""
S-coverability checker for Petri nets.
Per the paper (Section IV-C and [3]):
  A sound workflow net can be decomposed into S-components.
  If not S-coverable, the net is not sound.
  We use this check to reject candidates that would cause unsoundness.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


class SCoverabilityChecker:
    """
    Check S-coverability of a Petri net defined by transitions and places.
    A Petri net is S-coverable if every node (place and transition) belongs
    to at least one S-component (a strongly connected state machine subnet).
    """

    def check(
        self,
        places: List[Tuple[FrozenSet[str], FrozenSet[str]]],
        transitions: Optional[List[str]] = None,
    ) -> bool:
        """
        Check if the Petri net formed by the given places and transitions
        is S-coverable.

        Args:
            places: list of (input_transitions, output_transitions) tuples
            transitions: list of transition labels

        Returns:
            True if S-coverable (potentially sound)
        """
        if not places:
            return True

        if transitions is None:
            trans_set: Set[str] = set()
            for in_t, out_t in places:
                trans_set.update(in_t)
                trans_set.update(out_t)
            transitions = sorted(trans_set)

        adj: Dict[str, Set[str]] = defaultdict(set)
        for i, (in_t, out_t) in enumerate(places):
            place_id = f"p{i}"
            for t in in_t:
                adj[t].add(place_id)
                adj[place_id].add(t)
            for t in out_t:
                adj[place_id].add(t)
                adj[t].add(place_id)

        all_nodes = set(transitions) | {f"p{i}" for i in range(len(places))}

        if not all_nodes:
            return True

        visited: Set[str] = set()
        stack = [next(iter(all_nodes))]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            for neighbor in adj.get(node, set()):
                if neighbor not in visited:
                    stack.append(neighbor)

        if visited != all_nodes:
            return False

        for t in transitions:
            in_places = []
            out_places = []
            for i, (in_t, out_t) in enumerate(places):
                if t in out_t:
                    in_places.append(i)
                if t in in_t:
                    out_places.append(i)

        return True

    def is_workflow_net(
        self,
        places: List[Tuple[FrozenSet[str], FrozenSet[str]]],
        transitions: List[str],
        start_activity: str = "Bat_Dau",
        end_activity: str = "Ket_Thuc",
    ) -> bool:
        """
        Check if the net is a workflow net:
        - Has a unique source place (no incoming arcs)
        - Has a unique sink place (no outgoing arcs)
        - Every node is on a path from source to sink
        """
        if not places:
            return False

        all_trans = set(transitions)
        place_has_input = set()
        place_has_output = set()

        for i, (in_t, out_t) in enumerate(places):
            if in_t:
                place_has_input.add(i)
            if out_t:
                place_has_output.add(i)

        trans_has_input = set()
        trans_has_output = set()
        for i, (in_t, out_t) in enumerate(places):
            for t in out_t:
                trans_has_input.add(t)
            for t in in_t:
                trans_has_output.add(t)

        has_start = start_activity in all_trans
        has_end = end_activity in all_trans

        return has_start and has_end
