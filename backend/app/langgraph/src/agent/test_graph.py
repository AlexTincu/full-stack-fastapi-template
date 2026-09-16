"""Minimal chat graph, no retrieval — validates the langgraph dev <-> agent-chat-ui wiring."""

from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph

model = init_chat_model("openai:gpt-4.1-mini", temperature=0)


def call_model(state: MessagesState) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}


graph = (
    StateGraph(MessagesState)
    .add_node("call_model", call_model)
    .add_edge("__start__", "call_model")
    .compile(name="Test Agent")
)
