"""
Memory Agent: semantic search over past bills and transactions.
Standalone ADK agent; uses search_memory tool only.
"""

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from google.adk.agents import LlmAgent
from .tools import search_memory


memory_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="memory_agent",
    description="Searches and recalls past bills and transactions by natural language (e.g. 'internet bill', 'what I paid for electric').",
    instruction="""You are the memory specialist for TaskFin. You help users find and recall past bills and transactions.

When the user asks to find a bill, recall a payment, or search for something (e.g. 'internet bill', 'electricity payment', 'what did I pay for X'), use search_memory with their query. Present the results in a clear, readable way: show the text summary, amount, dates, and status where relevant.

If search_memory returns status 'error', tell the user that search is temporarily unavailable and suggest they try again later or list their bills from the finance section.

Keep responses concise and focused on the search results.""",
    tools=[search_memory],
)
