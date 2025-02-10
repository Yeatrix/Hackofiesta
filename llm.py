from typing import Annotated
from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
import os
from dotenv import load_dotenv

load_dotenv() 

class LLMModule:
    def __init__(self):
        self.memory = MemorySaver()
        self.graph = self._initialize_graph()
        
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    def _initialize_graph(self):
        # Initialize LLM and tools
        llm = ChatGroq(
            temperature=0.7,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile"
        )
        tool = TavilySearchResults(max_results=2, api_key=os.getenv("TRI_KEY"))
        llm_with_tools = llm.bind_tools([tool])

        # Build the graph
        graph_builder = StateGraph(self.State)
        
        # Add nodes
        graph_builder.add_node("chatbot", self.chatbot_node(llm_with_tools))
        graph_builder.add_node("tools", ToolNode(tools=[tool]))
        
        # Configure edges
        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.set_entry_point("chatbot")
        
        return graph_builder.compile()

    def chatbot_node(self, llm):
        def node_func(state: self.State):
            return {"messages": [llm.invoke(state["messages"])]}
        return node_func

    def generate_response(self, messages):
        config = {"configurable": {"thread_id": "1"}}
        for event in self.graph.stream({"messages": messages}, config):
            for value in event.values():
                return value["messages"][-1].content
        return None
