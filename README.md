# langGraph-playground

## LangGraph Complete Course for Beginners – Complex AI Agents with Python

Video Reference: https://youtu.be/jGg_1h0qzaM?si=izp7aHQTxQxdiajz

GitHub repo: https://github.com/iamvaibhavmehra/LangGraph-Course-freeCodeCamp/tree/main

## LangGraph
LangGraph is an open-source Python library created by LangChain designed to build robust, stateful, and complex multi-agent AI applications using LLMs.
- LangGraph is an orchestration framework built on top of LangChain.
- It uses a graph-based execution model, which makes it easier to implement complex, non-linear workflows.
- LangGraph includes helper constructs for loops and conditional branches that are not available in LangChain alone.
- Using LangGraph, you can create single-agent or multi-agent systems.

## LangGraph Core Components

LangGraph is an orchestration framework built on top of LangChain that uses a **graph-based execution model**, making it easier to implement complex, non-linear agentic workflows.

---

### 1. Nodes

A **node** is the fundamental unit of work in LangGraph. Each node represents a single task implemented as a Python function.

**Key characteristics:**
- Every node receives the current **state** as input
- Every node outputs an updated **state**
- Every node has full read and write access to the shared state
- Nodes can wrap LLM calls, tool invocations, or any custom logic

```python
def my_node(state: MyState) -> MyState:
    # Read from state, do work, return updated state
    result = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [result]}
```

---

### 2. Edges

**Edges** connect nodes and define the execution flow of the graph. They determine which node runs next after the current one completes.

**Types of edges:**

| Type | Description |
|------|-------------|
| Normal edge | Always transitions from node A to node B |
| Conditional edge | Routes to different nodes based on state values |
| Parallel paths | Fan out to multiple nodes simultaneously |

**Edges support:**
- Loops — a node can route back to a previous node
- Conditional branches — different paths based on logic
- Parallel execution — multiple nodes run at the same time

```python
# Normal edge
graph.add_edge("node_a", "node_b")

# Conditional edge
graph.add_conditional_edges(
    "router_node",
    route_function,  # returns name of next node
    {
        "path_a": "node_a",
        "path_b": "node_b",
    }
)
```

---

### 3. State

**State** is the shared memory that flows through the entire graph. It stores all data passed between nodes and maintains the execution context of the system.

**Key characteristics:**
- Defined as a `TypedDict` or a Pydantic model
- Every node takes state as input, modifies it, and returns the updated state
- Represented as key–value pairs
- Accessible and updatable by every node in the graph

```python
from typing import TypedDict, List

class MyState(TypedDict):
    messages: List[str]
    current_step: str
    result: str
```

**State acts as the single source of truth** — rather than passing data directly between nodes, all inter-node communication happens through the shared state object.

---

### 4. Reducers

**Reducers** define how updates to the state are applied when multiple nodes modify the same key. Without reducers, the default behavior is to overwrite the previous value.

#### Why Reducers Matter

Consider a state key called `messages`:

**Without a reducer (overwrites):**
```
Node 1 sets: messages → "Hi, my name is John"
Node 2 sets: messages → "Sup, how are you?"
→ Node 3 asks "What is my name?" — the system cannot answer because the first message was lost.
```

**With a reducer (appends):**
```
Node 1 sets: messages → ["Hi, my name is John"]
Node 2 sets: messages → ["Hi, my name is John", "Sup, how are you?"]
→ Node 3 can answer correctly because full history is maintained.
```

#### Reducer Strategies

| Strategy | Use case |
|----------|----------|
| Overwrite (default) | Single-value fields like `current_step`, `status` |
| Append | Conversation history, collected results |
| Merge | Dictionaries being built up incrementally |
| Custom logic | Domain-specific combining rules |

Each key in the state can have its own independent reducer.

```python
from typing import Annotated
from operator import add

class MyState(TypedDict):
    # This key will append instead of overwrite
    messages: Annotated[List[str], add]
    # This key will overwrite (default)
    status: str
```

---

### 5. Graph

The **StateGraph** is the container that ties all components together. You register nodes and edges on it, then compile it into a runnable graph.

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(MyState)

# Add nodes
graph.add_node("router", router_node)
graph.add_node("worker", worker_node)
graph.add_node("critic", critic_node)

# Add edges
graph.set_entry_point("router")
graph.add_conditional_edges("router", route_fn, {"worker": "worker", "end": END})
graph.add_edge("worker", "critic")
graph.add_edge("critic", END)

# Compile
app = graph.compile()
```

---

### 6. Checkpointer (Persistence)

A **checkpointer** saves the graph state after every node execution, enabling pause/resume, fault tolerance, and human-in-the-loop workflows.

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# Run with a thread_id to persist state across invocations
config = {"configurable": {"thread_id": "session-123"}}
result = app.invoke({"messages": ["start"]}, config=config)
```

**Benefits of checkpointing:**
- Resume from the exact point of failure
- Support long-running workflows across multiple sessions
- Enable human-in-the-loop interrupts and approvals
- Replay or branch execution history

---

### 7. MessageState (Prebuilt)

**MessageState** is a prebuilt state schema designed for conversational, chat-based agentic workflows. It removes the need to manually define state for chatbots.

```python
from langgraph.graph import MessageState, StateGraph

# MessageState already includes a `messages` key with the correct reducer
graph = StateGraph(MessageState)
```

It stores a list of typed messages (`HumanMessage`, `AIMessage`, `ToolMessage`, etc.) and handles appending automatically — no manual reducer configuration needed.

---

### Summary

| Component | Role |
|-----------|------|
| **Node** | A Python function that reads and updates state — the unit of work |
| **Edge** | Connects nodes; defines flow, branching, and loops |
| **State** | Shared key-value memory accessible by all nodes |
| **Reducer** | Controls how state keys are updated (overwrite, append, merge) |
| **Graph** | The container that compiles nodes + edges into a runnable workflow |
| **Checkpointer** | Persists state for fault tolerance, resume, and human-in-the-loop |
| **MessageState** | Prebuilt state schema for chat-based workflows |

---

### When to Use LangGraph

Use LangGraph when building:
- Agentic systems that need autonomous planning and execution
- Workflows with loops and conditional branching
- Multi-agent systems
- Human-in-the-loop approval flows
- Long-running, production-grade workflows that require checkpointing


## 📚 Repository Contents

This repository demonstrates LangGraph usage through interactive Jupyter notebooks and standalone scripts. It is organized by progressive exercises:

- `Agent_Bot.py`: interactive AI chatbot using LangGraph with Azure OpenAI integration. Features a simple state graph that processes user messages in a loop until "exit" is typed. Requires Azure OpenAI credentials in a `.env` file.
- `ReAct_Agent.py`: ReAct (Reasoning and Acting) agent implementation with tool usage. Demonstrates conditional edges, ToolNode integration, and the agent loop pattern where the model can call tools (add, subtract) and receive results before responding. Saves a Mermaid graph visualization to PNG.
- `Hello_World_Graph.ipynb`: first graph example, simple state update in one node
- `exercise_1.ipynb`: single node introduction with a compliment transformation
- `exercise_2.ipynb`: multi-input node (math add/multiply) plus operation routing
- `exercise_3.ipynb`: multi-node linear graph with message assembly from name, age, skills
- `exercise_4.ipynb`: conditional graph branching based on operator values (`+` or `-`)
- `exercise_5.ipynb`: looping logic via conditional edges (number guessing game)

## 🚀 Quick Start

1. Create and activate a Python environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install langgraph ipython langchain-openai python-dotenv
```

3. For `Agent_Bot.py`, create a `.env` file with your Azure OpenAI credentials:

```bash
AZURE_OPENAI_ENDPOINT=your-endpoint
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
```

4. Launch Jupyter:

```bash
jupyter notebook
```

5. Open any notebook and run cells sequentially.

## 🧩 Notes

- The notebooks use `StateGraph` from `langgraph.graph`.
- Each notebook includes a visual graph render (`Mermaid` PNG) and example invocation.
- `exercise_4` and `exercise_5` show conditional flows with `START`, `END`, and routing.

