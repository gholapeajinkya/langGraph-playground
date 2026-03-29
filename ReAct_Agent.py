# Reasoning and Acting (ReAct) Agent Implementation

from typing import Annotated, Sequence, TypedDict
# Annotated is used to provide type hints for the input and output of the process_node function
# Sequence is used to indicate that the messages input is a sequence of BaseMessage objects
# TypedDict is used to define the structure of the AgentState dictionary
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
# BaseMessage is the base class for all messages in the conversation
# ToolMessage is used to represent messages that involve tool usage, such as calling an API or executing a command
# SystemMessage is used to represent messages that provide instructions or context to the agent, such as the initial system prompt
from langgraph.graph.message import add_messages
# add_message is a utility function that can be used to add messages to the conversation history in a structured way (reducer function)
from langchain_openai import AzureChatOpenAI
from langchain_core.tools import tool
# tool is a decorator used to define a function as a tool that can be called by the agent during the conversation
from langgraph.prebuilt import ToolNode
# ToolNode is a prebuilt node that can be used to integrate tool usage into the state graph, allowing the agent to call tools as part of its reasoning and acting process
from dotenv import load_dotenv
import os

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # The AgentState dictionary has a single key "messages" which is a sequence of BaseMessage objects.
    # The add_message function is used as a reducer to manage the conversation history by adding new messages to the existing sequence.

# Tools


@tool
def add(a: int, b: int) -> int:
    """A simple tool that adds two numbers together."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """A simple tool that subtracts one number from another."""
    return a - b

tools = [add, subtract]

model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    # Bind tools to the model so it can use them during the conversation
).bind_tools(tools)


def process_node(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are a helpful assistant that can perform various tasks using tools. Use the provided tools to assist the user with their requests."
    )
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


# Conditional edge
def should_use_tool(state: AgentState) -> bool:
    last_message = state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"


graph = StateGraph(AgentState)
graph.add_node("Agent", process_node)

graph.add_node("Tools", ToolNode(tools=tools))
graph.add_edge(START, "Agent")

graph.add_conditional_edges(
    "Agent",
    should_use_tool,
    {
        "continue": "Tools",
        "end": END,
    },
)

graph.add_edge("Tools", "Agent")

app = graph.compile()

# Save the graph as a PNG image
# with open("react_agent_graph.png", "wb") as f:
#     f.write(app.get_graph().draw_mermaid_png())


def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


user_prompt = {"messages": [HumanMessage(content="What is 12 + 12, 89 + 152, and 123 + 456?")]}
# user_prompt = {"messages": [HumanMessage(content="Add 40 + 12 and then subtract 20 from the result. tell me a joke after that")]}

print_stream(app.stream(user_prompt, stream_mode="values"))
