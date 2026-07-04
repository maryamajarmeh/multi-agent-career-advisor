from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from schemas.interview import (
    InterviewQuestion,
    InterviewEvaluation,
)
from utils.prompt_loader import load_prompt

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.4,
)

# -----------------------------
# Question Generator
# -----------------------------

question_llm = llm.with_structured_output(
    InterviewQuestion
)

question_prompt = load_prompt(
    "interview_question_prompt.txt"
)

question_chain = question_prompt | question_llm


# -----------------------------
# Answer Evaluation
# -----------------------------

evaluation_llm = llm.with_structured_output(
    InterviewEvaluation
)

evaluation_prompt = load_prompt(
    "interview_evaluation_prompt.txt"
)

evaluation_chain = evaluation_prompt | evaluation_llm


# -----------------------------
# Interview Agent
# -----------------------------

class InterviewAgent:

    def generate_question(
        self,
        job_role: str,
        level: str = "Junior",
    ) -> InterviewQuestion:

        return question_chain.invoke(
            {
                "job_role": job_role,
                "level": level,
            }
        )

    def evaluate_answer(
        self,
        job_role: str,
        question: str,
        answer: str,
    ) -> InterviewEvaluation:

        return evaluation_chain.invoke(
            {
                "job_role": job_role,
                "question": question,
                "answer": answer,
            }
        )