"""Chat graph backed by the existing rag_simple retrieval + generation pipeline.

`langgraph dev` runs here using full-stack-fastapi-template/backend's own venv (see
langgraph.json's "dependencies": ["../.."]), where `app` — including `app.rag_simple` —
is already installed editable. No separate dependency set to maintain for this folder.
"""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph

from app.rag_simple.generation.answer import answer_question


def _to_history(messages: list) -> list[dict]:
    return [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in messages[:-1]
    ]


def call_rag(state: MessagesState) -> dict:
    question = state["messages"][-1].content
    history = _to_history(state["messages"])
    answer, _chunks = answer_question(question, history)
    return {"messages": [AIMessage(content=answer)]}


graph = (
    StateGraph(MessagesState)
    .add_node("call_rag", call_rag)
    .add_edge("__start__", "call_rag")
    .compile(name="RAG Agent (rag_simple)")
)
