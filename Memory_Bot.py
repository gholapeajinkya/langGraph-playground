from typing import TypedDict, List, Union
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


# Azure OpenAI configuration
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),    
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
)

def process_node(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}\n")
    state["messages"].append(AIMessage(content=response.content))
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process_node)
graph.add_edge(START, "process")
graph.add_edge("process", END)  
agent = graph.compile()

conversation_history: List[Union[HumanMessage, AIMessage]] = []

user_input = input("Enter: ")

while user_input.lower() != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})

    conversation_history = result["messages"] # This will allow chatbot to maintain the conversation history across turns

    user_input = input("Enter: ")