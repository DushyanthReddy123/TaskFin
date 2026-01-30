"""
Shared tool functions for TaskFin ADK agents.
These call the existing backend (DB and FAISS retriever).
"""

import sys
import os

# Ensure project root is on path so "backend" can be imported
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from backend.db import session as db_session
from backend.db import models


def get_user_info(user_id: int) -> dict:
    """
    Retrieves the current user's profile (name, email) by user ID.

    Use this tool when the user asks who they are, what their email is, or for account identity.

    Args:
        user_id: The unique identifier of the user (integer).

    Returns:
        A dictionary. On success: {"status": "success", "name": "...", "email": "..."}.
        On failure: {"status": "error", "error_message": "..."}.
    """
    db = db_session.SessionLocal()
    try:
        user = db.query(models.User).filter(models.User.id == user_id).first()
        if not user:
            return {"status": "error", "error_message": f"User with id {user_id} not found."}
        return {
            "status": "success",
            "name": user.name or "",
            "email": user.email or "",
            "user_id": user.id,
        }
    except Exception as e:
        return {"status": "error", "error_message": str(e)}
    finally:
        db.close()


def get_bills(user_id: int) -> dict:
    """
    Lists all bills for the given user (amount, due date, status).

    Use this when the user asks for their bills, what they owe, or bill summary.

    Args:
        user_id: The unique identifier of the user (integer).

    Returns:
        A dictionary. On success: {"status": "success", "bills": [{"id": ..., "name": ..., "amount": ..., "due_date": ..., "status": ...}, ...]}.
        On failure: {"status": "error", "error_message": "..."}.
    """
    db = db_session.SessionLocal()
    try:
        bills = db.query(models.Bill).filter(models.Bill.user_id == user_id).all()
        out = []
        for b in bills:
            out.append({
                "id": b.id,
                "name": b.name,
                "amount": float(b.amount),
                "due_date": b.due_date.isoformat() if b.due_date else None,
                "status": b.status,
            })
        return {"status": "success", "bills": out}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}
    finally:
        db.close()


def pay_bill(user_id: int, bill_id: int) -> dict:
    """
    Marks a specific bill as paid for the given user.

    Use this when the user asks to pay a bill or mark a bill as paid. Requires the bill id.

    Args:
        user_id: The unique identifier of the user (integer).
        bill_id: The unique identifier of the bill to mark as paid (integer).

    Returns:
        A dictionary. On success: {"status": "success", "bill": {"id": ..., "name": ..., "status": "paid", ...}}.
        On failure: {"status": "error", "error_message": "..."}.
    """
    db = db_session.SessionLocal()
    try:
        bill = db.query(models.Bill).filter(
            models.Bill.id == bill_id,
            models.Bill.user_id == user_id,
        ).first()
        if not bill:
            return {"status": "error", "error_message": f"Bill with id {bill_id} not found for this user."}
        bill.status = "paid"
        db.commit()
        db.refresh(bill)
        return {
            "status": "success",
            "bill": {
                "id": bill.id,
                "name": bill.name,
                "amount": float(bill.amount),
                "due_date": bill.due_date.isoformat() if bill.due_date else None,
                "status": bill.status,
            },
        }
    except Exception as e:
        db.rollback()
        return {"status": "error", "error_message": str(e)}
    finally:
        db.close()


def search_memory(query: str, k: int = 5) -> dict:
    """
    Searches past bills and transactions by natural language (e.g. "internet bill", "electricity payment").

    Use this when the user asks to find or recall a specific bill or transaction by description or topic.

    Args:
        query: The search query string (e.g. "internet bill", "electric payment").
        k: Number of results to return (default 5).

    Returns:
        A dictionary. On success: {"status": "success", "results": [{"text": "...", "metadata": {...}, "distance": ...}, ...]}.
        On failure: {"status": "error", "error_message": "..."}.
    """
    try:
        from backend import retriever
        hits = retriever.search(query, k=k)
        results = [
            {"text": h["text"], "metadata": h["metadata"], "distance": h["distance"], "rank": h["rank"]}
            for h in hits
        ]
        return {"status": "success", "results": results}
    except FileNotFoundError as e:
        return {"status": "error", "error_message": "Search index not available. Run seed_data.py first."}
    except Exception as e:
        return {"status": "error", "error_message": str(e)}
