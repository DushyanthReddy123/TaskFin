"""
Run a single TaskFin ADK agent standalone (auth, finance, or memory).
Usage: python -m agents.run_standalone --agent auth|finance|memory [--message "your message"] [--user-id 1]
Requires: GOOGLE_API_KEY in env, Postgres running (for auth/finance), and embeddings (for finance/memory).
"""

import argparse
import asyncio
import os
import sys

# Load .env from project root so GOOGLE_API_KEY is available
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
    load_dotenv(_env_path)
except ImportError:
    pass

# Project root on path for backend and agents
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


APP_NAME = "taskfin_standalone"
USER_ID = "standalone_user"
SESSION_ID = "standalone_session"


def get_agent(agent_name: str):
    if agent_name == "auth":
        from backend.agents.auth_agent import auth_agent
        return auth_agent
    if agent_name == "finance":
        from backend.agents.finance_agent import finance_agent
        return finance_agent
    if agent_name == "memory":
        from backend.agents.memory_agent import memory_agent
        return memory_agent
    raise ValueError(f"Unknown agent: {agent_name}. Use auth, finance, or memory.")


def get_sample_message(agent_name: str) -> str:
    if agent_name == "auth":
        return "Who am I? My user_id is 1."
    if agent_name == "finance":
        return "List my bills. My user_id is 1."
    if agent_name == "memory":
        return "Find my internet bill."
    return "Hello."


async def run_agent(agent_name: str, message: str, user_id: int | None) -> None:
    agent = get_agent(agent_name)
    session_service = InMemorySessionService()
    # Pass user_id in session state so agent instructions can use {user:user_id?}
    initial_state = {}
    if user_id is not None:
        initial_state["user:user_id"] = user_id
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
        state=initial_state,
    )

    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    content = types.Content(role="user", parts=[types.Part(text=message)])
    sid = getattr(session, "id", None) or SESSION_ID

    print(f"[{agent_name}_agent] Query: {message}\n")
    final_text = None
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=sid,
        new_message=content,
    ):
        if getattr(event, "is_final_response", lambda: False)():
            if getattr(event, "content", None) and getattr(event.content, "parts", None):
                part = event.content.parts[0] if event.content.parts else None
                if part and getattr(part, "text", None):
                    final_text = part.text
    if final_text:
        print(f"[{agent_name}_agent] Response:\n{final_text}")
    else:
        print(f"[{agent_name}_agent] No final response captured.")


def main():
    parser = argparse.ArgumentParser(description="Run TaskFin ADK agent standalone.")
    parser.add_argument(
        "--agent",
        choices=["auth", "finance", "memory"],
        required=True,
        help="Which agent to run",
    )
    parser.add_argument(
        "--message",
        default=None,
        help="User message (default: sample message for the agent)",
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=1,
        help="user_id for auth/finance tools (default: 1)",
    )
    args = parser.parse_args()

    message = args.message or get_sample_message(args.agent)
    user_id = args.user_id if args.agent in ("auth", "finance") else None

    if not os.environ.get("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY not set. Set it in .env or environment for Gemini.")
    asyncio.run(run_agent(args.agent, message, user_id))


if __name__ == "__main__":
    main()
