"""
Auth Agent: handles identity and login-related questions.
Standalone ADK agent; use get_user_info tool to answer "who am I?" etc.
"""

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from google.adk.agents import LlmAgent
from .tools import get_user_info


auth_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="auth_agent",
    description="Handles user identity and login-related questions. Use for 'who am I?', account info, or how to log in.",
    instruction="""You are the identity and auth specialist for TaskFin.

The current user_id for this session is {user:user_id?}. Use this value when calling get_user_info.

When the user asks who they are, what their email is, or for account info, use the get_user_info tool with the user_id above. Then respond with their name and email in a friendly way.

If the user asks how to log in, explain that they can use the app's login endpoint with their email and password to receive a token. Do not call get_user_info for login instructions.

If get_user_info returns status 'error', tell the user the account was not found and suggest they log in or check the user_id.

Keep responses concise and helpful.""",
    tools=[get_user_info],
)
