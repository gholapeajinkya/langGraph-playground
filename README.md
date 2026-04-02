# langGraph-playground

LangGraph Complete Course for Beginners – Complex AI Agents with Python

Video Reference: https://youtu.be/jGg_1h0qzaM?si=izp7aHQTxQxdiajz

Github repo: https://github.com/iamvaibhavmehra/LangGraph-Course-freeCodeCamp/tree/main

## Langgraph
LangGraph is an open-source Python library created by LangChain designed to build robust, stateful, and complex multi-agent AI applications using LLMs

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

4. Open any notebook and run cells sequentially.

## 🧩 Notes

- The notebooks use `StateGraph` from `langgraph.graph`.
- Each notebook includes a visual graph render (`Mermaid` PNG) and example invocation.
- `exercise_4` and `exercise_5` show conditional flows with `START`, `END`, and routing.

## ✅ Validation

All notebooks are reviewed and accurately described in this README. If you add new exercise notebooks, append them here with the same summary format.
