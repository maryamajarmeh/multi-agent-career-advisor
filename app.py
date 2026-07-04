from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import hashlib
import json
import time
import uuid

from coordinator import CoordinatorAgent
from memory import save_user_memory, get_user_memory
from logger import logger
from traces import save_trace

from metrics import REQUEST_COUNT, ERROR_COUNT, LATENCY
from prometheus_client import generate_latest
from fastapi.responses import Response

# 🔐 PII MASKING (IMPORTANT)
from utils.pii_masking import mask_pii

app = FastAPI()
coordinator = CoordinatorAgent()


class ChatRequest(BaseModel):
    user_id: str
    message: str
    resume: str | None = None
    level: str | None = "Junior"


@app.post("/chat")
def chat(data: ChatRequest):

    REQUEST_COUNT.inc()

    start = time.time()
    session_id = str(uuid.uuid4())
    steps = []

    try:
        steps.append("Receive Message")

        # 🔐 PII protection
        safe_message = mask_pii(data.message or "")
        safe_resume = mask_pii(data.resume or "")

        memory = get_user_memory(data.user_id)
        steps.append("Load Memory")

        user_data = {
            "resume": safe_resume,
            "career_goal": memory.get("goal"),
            "level": data.level,
            # lets the coordinator know a question is still awaiting an
            # answer, so it evaluates this message instead of asking a
            # brand new question
            "pending_interview_question": memory.get("pending_interview_question"),
        }

        # 🔥 Coordinator execution
        agent_result = coordinator.run(
            message=safe_message,
            user_data=user_data
        )

        steps.append("Coordinator Execution")

        # -----------------------------
        # Normalize coordinator output
        # -----------------------------
        # IMPORTANT: we do NOT flatten structured replies (dict/list) into
        # strings anymore. The frontend (Streamlit) knows how to render
        # dicts/lists nicely via ui_components.render_agent_response().
        # Flattening here was the root cause of "ugly raw dict" output.
        # _UNSET tells apart "the interview agent didn't run this turn"
        # from "the interview agent ran and explicitly cleared the
        # pending question" (a real None).
        _UNSET = object()

        if isinstance(agent_result, dict):
            reply = agent_result.get("response", "")
            agent_used = agent_result.get("agent_used", "coordinator")
            pending_interview_question = agent_result.get(
                "pending_interview_question", _UNSET
            )
        else:
            reply = str(agent_result)
            agent_used = "coordinator"
            pending_interview_question = _UNSET

        # Text-only version, safe to persist to memory/logs regardless of
        # whether `reply` is a string, dict, or list.
        if isinstance(reply, str):
            reply_text_for_memory = reply
        else:
            try:
                reply_text_for_memory = json.dumps(reply, ensure_ascii=False)
            except TypeError:
                reply_text_for_memory = str(reply)

        # Save memory
        memory_updates = {
            "last_message": safe_message,
            "last_response": reply_text_for_memory,
            "timestamp": str(datetime.now())
        }

        # Only touch interview state when the interview agent actually ran
        # this turn, so asking a career/resume question mid-interview
        # doesn't silently wipe out the question the user still owes an
        # answer to.
        if pending_interview_question is not _UNSET:
            memory_updates["pending_interview_question"] = pending_interview_question

        # Append this Q&A pair to the running interview history, so
        # /end_interview can build a final report across the whole session.
        history_entry = agent_result.get("interview_history_entry") if isinstance(agent_result, dict) else None
        if history_entry is not None:
            interview_history = memory.get("interview_history", []) or []
            interview_history = list(interview_history) + [history_entry]
            memory_updates["interview_history"] = interview_history

        save_user_memory(data.user_id, memory_updates)

        steps.append("Save Memory")

        latency = time.time() - start
        LATENCY.observe(latency)

        logger.info({
            "timestamp": str(datetime.now()),
            "request_id": session_id,
            "hashed_user_id": hashlib.sha256(data.user_id.encode()).hexdigest(),
            "prompt": safe_message,
            "model_version": "gpt-4.1-mini",
            "latency_ms": round(latency * 1000),
            "error_code": None
        })

        save_trace(session_id, steps)

        return {
            "session_id": session_id,
            "response": reply,          # kept as native dict/list/str
            "agent_used": agent_used,
            "memory": memory
        }

    except Exception as e:

        ERROR_COUNT.inc()

        logger.error({
            "timestamp": str(datetime.now()),
            "request_id": session_id,
            "error": str(e)
        })

        return {
            "error": str(e),
            "session_id": session_id
        }


class EndInterviewRequest(BaseModel):
    user_id: str


@app.post("/end_interview")
def end_interview(data: EndInterviewRequest):
    """Ends the current interview session on demand and returns a final,
    aggregated evaluation across every question that was answered."""
    try:
        memory = get_user_memory(data.user_id)
        history = memory.get("interview_history", []) or []

        final_summary = coordinator.summarize_interview(history)

        # Reset interview state so the next question starts a fresh session
        save_user_memory(
            data.user_id,
            {
                "pending_interview_question": None,
                "interview_history": [],
            },
        )

        return {
            "response": final_summary,
            "agent_used": "interview",
        }

    except Exception as e:
        logger.error({
            "timestamp": str(datetime.now()),
            "error": str(e),
            "context": "end_interview",
        })
        return {"error": str(e)}