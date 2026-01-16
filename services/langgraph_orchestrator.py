
"""Lightweight event graph service backed by NetworkX.

This module provides a small, well-typed API over a directed graph that
stores events and simple metadata. It includes safe persistence helpers
and clear error handling suitable for unit testing and productionization.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path
import json
import threading
import logging

import networkx as nx

logger = logging.getLogger(__name__)


class EventGraphService:
    """Manage a directed event graph with small helper utilities.

    Contract:
    - add_event: returns a stable integer node id
    - add_edge: creates a directed edge between existing node ids
    - save / load: persist a JSON node-link representation
    """

    def __init__(self, graph: Optional[nx.DiGraph] = None) -> None:
        self._lock = threading.RLock()
        self.graph: nx.DiGraph = graph if graph is not None else nx.DiGraph()

    def add_event(self, text: str, meta: Optional[Dict[str, Any]] = None) -> int:
        """Add an event node and return its integer id.

        Args:
            text: Short textual representation of the event.
            meta: Optional metadata stored as a dict.
        Returns:
            int: node id allocated for the event.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        with self._lock:
            idx = len(self.graph.nodes)
            self.graph.add_node(idx, text=text, meta=meta or {})
            logger.debug("Added event node %s: %s", idx, text)
            return idx

    def add_edge(self, src: int, dst: int, weight: float = 1.0) -> None:
        """Add a directed edge between two node ids.

        Raises:
            KeyError: if either src or dst does not exist in the graph.
        """
        with self._lock:
            if src not in self.graph.nodes or dst not in self.graph.nodes:
                raise KeyError("Both src and dst must be existing node ids")
            self.graph.add_edge(src, dst, weight=float(weight))
            logger.debug("Added edge %s -> %s (weight=%s)", src, dst, weight)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable representation of the graph."""
        with self._lock:
            return nx.node_link_data(self.graph)

    def save(self, path: str | Path) -> None:
        """Persist the graph as a JSON file.

        The parent directory is created if necessary.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Graph saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "EventGraphService":
        """Load a graph from a file previously written by `save`.

        Raises:
            FileNotFoundError: if the path does not exist.
            ValueError: if the file cannot be parsed into a node-link graph.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        try:
            graph = nx.node_link_graph(payload)
        except Exception as exc:  # pragma: no cover - parsing error
            logger.exception("Failed to parse graph JSON")
            raise ValueError("invalid graph file") from exc
        logger.info("Graph loaded from %s", path)
        return cls(graph=graph)

