import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Writing the final answer: the strongest model. Grading evidence is a simpler
# yes/no judgment, so a smaller, faster model is enough.
# Override either in .env; see https://console.groq.com/docs/models
ANSWER_MODEL = os.getenv("GROQ_ANSWER_MODEL", "openai/gpt-oss-120b")
GRADER_MODEL = os.getenv("GROQ_GRADER_MODEL", "openai/gpt-oss-20b")


@lru_cache
def get_llm(model: str = ANSWER_MODEL) -> ChatGroq:
    """Groq chat model (temperature 0 so answers and grades are repeatable)"""
    return ChatGroq(model=model, temperature=0, max_retries=2)


def get_grader_llm() -> ChatGroq:
    return get_llm(GRADER_MODEL)
