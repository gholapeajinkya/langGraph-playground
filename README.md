# langGraph-playground

LangGraph Complete Course for Beginners – Complex AI Agents with Python

Video Reference: https://youtu.be/jGg_1h0qzaM?si=izp7aHQTxQxdiajz

## 📚 Repository Contents

This repository demonstrates LangGraph usage through interactive Jupyter notebooks. It is organized by progressive exercises:

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
pip install langgraph ipython
```

3. Launch Jupyter:

```bash
jupyter notebook
```

4. Open any notebook and run cells sequentially.

## 🧩 Notes

- The notebooks use `StateGraph` from `langgraph.graph`.
- Each notebook includes a visual graph render (`Mermaid` PNG) and example invocation.
- `exercise_4` and `exercise_5` show conditional flows with `START`, `END`, and routing.

## ✅ Validation

All notebooks are reviewed and accurately described in this README. If you add new exercise notebooks, append them here with the same summary format.
