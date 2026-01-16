import json
from pathlib import Path

import pytest

from services.langgraph_orchestrator import EventGraphService


def test_add_nodes_and_edges(tmp_path: Path):
    svc = EventGraphService()
    a = svc.add_event("Event A", meta={"source": "test"})
    b = svc.add_event("Event B")
    assert isinstance(a, int) and isinstance(b, int)
    svc.add_edge(a, b, weight=2.5)
    # save and load
    out = tmp_path / "graph.json"
    svc.save(out)
    assert out.exists()

    loaded = EventGraphService.load(out)
    # nodes preserved
    assert loaded.graph.number_of_nodes() == 2
    assert loaded.graph.number_of_edges() == 1
    assert loaded.graph.nodes[a]["text"] == "Event A"


def test_add_edge_invalid_nodes():
    svc = EventGraphService()
    with pytest.raises(KeyError):
        svc.add_edge(0, 1)
