from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from schemas.router import RouterDecision

from agents.career_agent import CareerAgent
from agents.resume_agent import ResumeAgent
from agents.interview_agent import InterviewAgent

load_dotenv()


# -------------------------
# Router LLM
# -------------------------

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)

router_llm = llm.with_structured_output(RouterDecision)

router_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a routing system for an AI Career Coach.

Decide which agent should handle the user's request.

Available agents:

- career
  Questions about career advice, roadmap, learning plan,
  certifications, required skills, or career guidance.

- resume
  Resume review, ATS score, CV improvement, resume writing,
  resume feedback.

- interview
  Interview preparation, interview questions,
  mock interviews, interview answer evaluation.

Return the appropriate agent.
            """
        ),
        ("human", "{message}"),
    ]
)

router_chain = router_prompt | router_llm


# -------------------------
# Phrases that mean "I don't want to answer this one, give me a
# different question" instead of "here is my answer".
# -------------------------
INTERVIEW_SKIP_PHRASES = [
    "new question", "another question", "next question", "different question",
    "skip this", "skip question", "give me another", "ask me something else",
    "سؤال جديد", "سؤال تاني", "سؤال آخر", "سؤال مختلف", "تخطي",
]


# -------------------------
# Coordinator Agent
# -------------------------

class CoordinatorAgent:

    def __init__(self):
        self.career_agent = CareerAgent()
        self.resume_agent = ResumeAgent()
        self.interview_agent = InterviewAgent()

    def route(self, message: str) -> str:
        decision = router_chain.invoke(
            {"message": message}
        )
        return decision.agent

    @staticmethod
    def _extract_question_text(result) -> str:
        """Pulls the plain question string out of whatever the
        interview agent's structured output looks like."""
        if hasattr(result, "question"):
            return result.question
        if isinstance(result, dict):
            return (
                result.get("question")
                or result.get("interview_question")
                or str(result)
            )
        return str(result)

    @staticmethod
    def _to_plain(obj):
        """Converts a pydantic model (or anything else) into a plain
        JSON-safe dict, for storing in memory."""
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "dict"):
            return obj.dict()
        return obj

    @staticmethod
    def _field(obj, *names):
        for name in names:
            if isinstance(obj, dict) and obj.get(name):
                return obj.get(name)
            if hasattr(obj, name) and getattr(obj, name):
                return getattr(obj, name)
        return None

    def summarize_interview(self, history: list) -> dict:
        """Aggregates all evaluated Q&A pairs from an interview session
        into one final report. Pure Python — no extra LLM call needed."""
        if not history:
            return {
                "message": "No interview questions were answered in this session yet.",
            }

        scores = []
        strengths_all, weaknesses_all, missing_all = [], [], []

        for entry in history:
            evaluation = entry.get("evaluation") if isinstance(entry, dict) else None
            if not evaluation:
                continue

            score = self._field(evaluation, "score")
            if score is not None:
                try:
                    scores.append(float(score))
                except (TypeError, ValueError):
                    pass

            strengths = self._field(evaluation, "strengths") or []
            weaknesses = self._field(evaluation, "weaknesses") or []
            missing = self._field(evaluation, "missing_points", "missing")

            if isinstance(strengths, list):
                strengths_all.extend(strengths)
            if isinstance(weaknesses, list):
                weaknesses_all.extend(weaknesses)
            if isinstance(missing, list):
                missing_all.extend(missing)

        def _dedupe(items, limit=6):
            seen = []
            for item in items:
                if item not in seen:
                    seen.append(item)
            return seen[:limit]

        average_score = round(sum(scores) / len(scores), 1) if scores else None

        return {
            "questions_answered": len(history),
            "average_score": average_score,
            "score_out_of": 10,
            "top_strengths": _dedupe(strengths_all),
            "areas_to_improve": _dedupe(weaknesses_all + missing_all),
        }

    def run(
        self,
        message: str,
        user_data: dict | None = None,
    ):

        user_data = user_data or {}

        pending_question = user_data.get("pending_interview_question")
        wants_new_question = bool(pending_question) and any(
            phrase in message.lower() for phrase in INTERVIEW_SKIP_PHRASES
        )

        # -------------------------
        # STICKY INTERVIEW FLOW
        # -------------------------
        # If the candidate was just asked a question, treat this message
        # as their answer, evaluate it, AND immediately ask the next
        # question in the same turn — the interview keeps going on its
        # own until the candidate explicitly ends it (via the "End
        # Interview" action, handled in app.py).
        if pending_question and not wants_new_question:
            evaluation_result = self.interview_agent.evaluate_answer(
                job_role=user_data.get("career_goal", "AI Engineer"),
                question=pending_question,
                answer=message,
            )

            next_question_result = self.interview_agent.generate_question(
                job_role=user_data.get("career_goal", "AI Engineer"),
                level=user_data.get("level", "Junior"),
            )
            next_question_text = self._extract_question_text(next_question_result)

            return {
                "response": {
                    "evaluation": evaluation_result,
                    "next_question": next_question_result,
                },
                "agent_used": "interview",
                # the interview keeps going: this new question is now pending
                "pending_interview_question": next_question_text,
                # so app.py can append this Q&A pair to the session history
                "interview_history_entry": {
                    "question": pending_question,
                    "evaluation": self._to_plain(evaluation_result),
                },
            }

        route = self.route(message)

        # -------------------------
        # CAREER AGENT
        # -------------------------
        if route == "career":
            result = self.career_agent.get_career_advice(
                goal=message
            )

            return {
                "response": result,
                "agent_used": "career"
            }

        # -------------------------
        # RESUME AGENT
        # -------------------------
        elif route == "resume":
            result = self.resume_agent.review_resume(
                resume=user_data.get("resume", ""),
                career_goal=user_data.get("career_goal", "AI Engineer"),
            )

            return {
                "response": result,
                "agent_used": "resume"
            }

        # -------------------------
        # INTERVIEW AGENT (new question / start of interview)
        # -------------------------
        elif route == "interview":
            result = self.interview_agent.generate_question(
                job_role=user_data.get("career_goal", "AI Engineer"),
                level=user_data.get("level", "Junior"),
            )

            return {
                "response": result,
                "agent_used": "interview",
                # remember this question so the NEXT message is treated
                # as the candidate's answer instead of a fresh request
                "pending_interview_question": self._extract_question_text(result),
            }

        # -------------------------
        # fallback
        # -------------------------
        return {
            "response": "Sorry, I couldn't determine the correct agent.",
            "agent_used": "router_fallback"
        }