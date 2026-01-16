
from fastapi import FastAPI
from services.langgraph_orchestrator import EventGraphService

app = FastAPI(title="FERS API")
graph = EventGraphService()

@app.post("/predict")
def predict():
    e1 = graph.add_event("Rate hike detected")
    e2 = graph.add_event("Liquidity tightens")
    graph.add_edge(e1,e2)
    graph.save("event_graph.json")
    return {"forecast":"downtrend","explanation":"Rate hike reduces liquidity"}
