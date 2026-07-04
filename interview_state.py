"""
Tracks a per-user interview SESSION (not just a single pending question),
so the coordinator can:
  1) evaluate an answer
  2) immediately generate the next question
  3) keep looping until the user explicitly ends the session

Swap the dict for Redis / a DB table in production — the interface is
what matters, not the storage backend.
"""

import threading
import time

_lock = threading.Lock()
_sessions: dict[str, dict] = {}

TTL_SECONDS = 60 * 30  # abandon a stale session after 30 min of inactivity

# Sent by the UI as a literal, exact-match control message (never inferred
# from natural language) so a real answer can never accidentally end the
# interview just because it happens to contain a word like "stop" or "done".
END_INTERVIEW_SENTINEL = "__END_INTERVIEW__"


def _get(user_id: str) -> dict | None:
    session = _sessions.get(user_id)
    if session is None:
        return None
    if time.time() - session["updated_at"] > TTL_SECONDS:
        del _sessions[user_id]
        return None
    return session


def start_session(user_id: str, job_role: str, level: str) -> None:
    with _lock:
        _sessions[user_id] = {
            "job_role": job_role,
            "level": level,
            "pending_question": None,
            "history": [],  # [{"question": str, "answer": str, "score": int}]
            "updated_at": time.time(),
        }


def has_active_session(user_id: str) -> bool:
    with _lock:
        return _get(user_id) is not None


def get_session_context(user_id: str) -> dict | None:
    with _lock:
        session = _get(user_id)
        if session is None:
            return None
        return {"job_role": session["job_role"], "level": session["level"]}


def set_pending_question(user_id: str, question: str) -> None:
    with _lock:
        session = _get(user_id)
        if session is None:
            return
        session["pending_question"] = question
        session["updated_at"] = time.time()


def get_pending_question(user_id: str) -> str | None:
    with _lock:
        session = _get(user_id)
        if session is None:
            return None
        return session.get("pending_question")


def record_answer(user_id: str, question: str, answer: str, score: int) -> None:
    with _lock:
        session = _get(user_id)
        if session is None:
            return
        session["history"].append({"question": question, "answer": answer, "score": score})
        session["pending_question"] = None
        session["updated_at"] = time.time()


def end_session(user_id: str) -> dict | None:
    """Clears the session and returns a summary, or None if there was none."""
    with _lock:
        session = _sessions.pop(user_id, None)
        if session is None:
            return None

        history = session["history"]
        avg_score = round(sum(h["score"] for h in history) / len(history), 2) if history else 0

        return {
            "job_role": session["job_role"],
            "level": session["level"],
            "questions_answered": len(history),
            "average_score": avg_score,
            "history": history,
        }
