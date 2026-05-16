"""Sample 3-agent LangGraph customer support system.

This is a FIXTURE — the system under test for Dry Run's development.
Not part of the product. Not extensible. Time-boxed.

Agents:
  - triage_agent: routes to sales or support
  - sales_agent: handles inventory search, cart, checkout
  - support_agent: handles order status, refunds

Tools:
  - search_inventory(query: str) -> list[dict]
  - add_to_cart(item_id: str, quantity: int) -> dict
  - update_cart(item_id: str, quantity: int) -> dict
  - process_checkout(cart_id: str) -> dict
  - check_order_status(order_id: str) -> dict
  - initiate_refund(order_id: str, reason: str) -> dict
"""

from __future__ import annotations
import os
import operator
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


# --- State ---


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_agent: str


# --- Tools ---


@tool
def search_inventory(query: str) -> list[dict]:
    """Search the product inventory."""
    return [
        {"id": "laptop-001", "name": "ThinkPad T14", "price": 899},
        {"id": "laptop-002", "name": "MacBook Air M3", "price": 1099},
        {"id": "tablet-001", "name": "iPad Air", "price": 599},
    ]


@tool
def add_to_cart(item_id: str, quantity: int = 1) -> dict:
    """Add an item to the cart."""
    return {"status": "added", "item_id": item_id, "quantity": quantity}


@tool
def update_cart(item_id: str, quantity: int) -> dict:
    """Update quantity of an item in the cart."""
    return {"status": "updated", "item_id": item_id, "quantity": quantity}


@tool
def process_checkout(cart_id: str = "default") -> dict:
    """Process checkout for the current cart."""
    return {"status": "confirmed", "order_id": "ORD-12345", "total": 899}


@tool
def check_order_status(order_id: str) -> dict:
    """Check the status of an order."""
    return {"order_id": order_id, "status": "shipped", "eta": "2 days"}


@tool
def initiate_refund(order_id: str, reason: str) -> dict:
    """Initiate a refund for an order."""
    return {"order_id": order_id, "refund_status": "initiated", "reason": reason}


# --- Agent nodes ---

sales_tools = [search_inventory, add_to_cart, update_cart, process_checkout]
support_tools = [check_order_status, initiate_refund]

_llm = None


def _get_llm():
    """Lazy-init the LLM based on environment. Defaults to Anthropic."""
    global _llm
    if _llm is not None:
        return _llm

    provider = os.environ.get("DRYRUN_LLM_PROVIDER", "anthropic")
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        model = os.environ.get("DRYRUN_AGENT_MODEL", "claude-sonnet-4-20250514")
        _llm = ChatAnthropic(model=model, temperature=0)
    else:
        from langchain_openai import ChatOpenAI

        model = os.environ.get("DRYRUN_AGENT_MODEL", "gpt-4o-mini")
        _llm = ChatOpenAI(model=model, temperature=0)
    return _llm


def triage_agent(state: AgentState) -> dict:
    """Route to sales or support based on user intent."""
    messages = state["messages"]
    llm = _get_llm()
    response = llm.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are a triage agent. Based on the user's message, decide:\n"
                    "- If about purchasing, browsing, cart: respond with 'ROUTE:sales'\n"
                    "- If about order status, refunds, issues: respond with 'ROUTE:support'\n"
                    "- Otherwise: respond helpfully and ask for clarification."
                ),
            },
            *messages,
        ]
    )
    return {"messages": [response], "current_agent": "triage"}


def sales_agent(state: AgentState) -> dict:
    """Handle sales-related queries."""
    messages = state["messages"]
    llm = _get_llm()
    response = llm.bind_tools(sales_tools).invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are a sales agent. Help the customer find products, "
                    "manage their cart, and complete purchases. Be helpful and concise."
                ),
            },
            *messages,
        ]
    )
    return {"messages": [response], "current_agent": "sales"}


def support_agent_node(state: AgentState) -> dict:
    """Handle support-related queries."""
    messages = state["messages"]
    llm = _get_llm()
    response = llm.bind_tools(support_tools).invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are a customer support agent. Help with order status, "
                    "refunds, and issues. Be empathetic and efficient."
                ),
            },
            *messages,
        ]
    )
    return {"messages": [response], "current_agent": "support"}


# --- Routing ---


def route_from_triage(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    if "ROUTE:sales" in content:
        return "sales_agent"
    elif "ROUTE:support" in content:
        return "support_agent"
    return END


def should_use_tools(state: AgentState) -> str:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END


# --- Graph ---

sales_tool_node = ToolNode(sales_tools)
support_tool_node = ToolNode(support_tools)

graph = StateGraph(AgentState)

graph.add_node("triage", triage_agent)
graph.add_node("sales_agent", sales_agent)
graph.add_node("support_agent", support_agent_node)
graph.add_node("sales_tools", sales_tool_node)
graph.add_node("support_tools", support_tool_node)

graph.set_entry_point("triage")

graph.add_conditional_edges(
    "triage",
    route_from_triage,
    {
        "sales_agent": "sales_agent",
        "support_agent": "support_agent",
        END: END,
    },
)

graph.add_conditional_edges(
    "sales_agent",
    should_use_tools,
    {
        "tools": "sales_tools",
        END: END,
    },
)

graph.add_conditional_edges(
    "support_agent",
    should_use_tools,
    {
        "tools": "support_tools",
        END: END,
    },
)

graph.add_edge("sales_tools", "sales_agent")
graph.add_edge("support_tools", "support_agent")

# Compile with MemorySaver for state isolation via thread_id
compiled_graph = graph.compile(checkpointer=MemorySaver())
