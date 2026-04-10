"""
Generate 20 unique IT process graph templates (G3-G22).

Each template has a genuinely different topology, activities, and flow
logic — covering CI/CD, security, infrastructure, data engineering, etc.
"""
from __future__ import annotations

import random
from typing import Dict

from .graph_reader import ProcessGraph


class GraphGenerator:
    """Generate diverse process graphs from built-in templates."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_variants(
        self,
        base_graphs: Dict[str, ProcessGraph],
        num_variants: int = 20,
    ) -> Dict[str, ProcessGraph]:
        """Return up to *num_variants* unique process graph templates (G3-G22).

        Each template has a completely different topology and activity set.
        *base_graphs* (G1, G2 from CSV) are not modified.
        """
        from .process_templates import create_diverse_templates

        all_templates = create_diverse_templates(seed=self.rng.randint(0, 2**31))
        sorted_keys = sorted(all_templates.keys(), key=lambda g: int(g[1:]))
        return {k: all_templates[k] for k in sorted_keys[:num_variants]}
