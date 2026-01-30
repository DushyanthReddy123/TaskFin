"""
Finance Agent: handles bills and payments.
Standalone ADK agent; uses get_bills, pay_bill (and optionally search) tools.
"""

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from google.adk.agents import LlmAgent
from .tools import get_bills, pay_bill, search_memory


finance_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="finance_agent",
    description="Handles bills and payments: list bills, mark bill as paid, and search bills/transactions by description.",
    instruction="""You are the finance specialist for TaskFin.

The current user_id for this session is {user:user_id?}. Use this value when calling get_bills or pay_bill.

When the user asks for their bills, what they owe, or a list of bills, use get_bills with the user_id above. Summarize the bills: name, amount, due date, and status (paid/unpaid).

When the user asks to pay a bill or mark a bill as paid, use pay_bill with user_id and bill_id. Confirm the bill name and that it was marked as paid.

When the user asks to find a bill by description (e.g. 'internet bill', 'electric bill'), use search_memory with their query to get relevant bills/transactions, then present the results clearly.

If any tool returns status 'error', tell the user what went wrong and suggest they check the user_id or bill_id.

Keep responses clear and actionable.""",
    tools=[get_bills, pay_bill, search_memory],
)
