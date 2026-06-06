"""
generator.py
------------
Phase 5: Generating grounded answers using Groq API with streaming.

What this file does:
1. Takes the retrieved context + user question
2. Builds a grounded prompt
3. Calls Groq API (Llama 3)
4. Streams the response token by token
"""

import os
from typing import Generator as TypingGenerator

from groq import Groq
from dotenv import load_dotenv

load_dotenv()


# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

GROQ_MODEL = "llama-3.1-8b-instant"
# GROQ_MODEL = "llama-3.3-70b-versatile"

MAX_TOKENS = 1024
TEMPERATURE = 0

# Safety limit to avoid sending excessive context
MAX_CONTEXT_CHARS = 12000


# ──────────────────────────────────────────────
# PROMPT TEMPLATE
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise document question-answering assistant.

Your job is to answer questions ONLY using the provided context from the eBay User Agreement document.

Rules:
- Answer ONLY based on the provided context.
- Do NOT use any outside knowledge.
- If the answer is not in the context, say exactly:
  "I could not find this information in the provided document."
- Be concise and accurate.
- Do not speculate or make assumptions.
- Quote or reference specific sections when possible.
"""


def build_user_prompt(context: str, question: str) -> str:
    """
    Constructs the user-facing prompt with injected context.
    """

    return f"""
====================
DOCUMENT CONTEXT
====================

{context}

====================
QUESTION
====================

{question}

====================
INSTRUCTIONS
====================

- Answer only using the document context above.
- Do not use outside knowledge.
- If the answer is not explicitly present in the context,
  respond exactly:
  "I could not find this information in the provided document."

====================
ANSWER
====================
"""


# ──────────────────────────────────────────────
# GENERATOR CLASS
# ──────────────────────────────────────────────

class Generator:
    """
    Handles LLM generation via Groq API.
    Supports both streaming and non-streaming modes.
    """

    def __init__(self, model: str = GROQ_MODEL):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found.\n"
                "Please add GROQ_API_KEY=your_key_here to your .env file.\n"
                "Get your free key from the Groq website."
            )

        self.client = Groq(api_key=api_key)
        self.model = model

        print(f"Generator ready. Model: {self.model}")

    def stream_answer(
        self,
        context: str,
        question: str
    ) -> TypingGenerator[str, None, None]:
        """
        Streams the LLM response token by token.

        This is a Python GENERATOR function — it yields text chunks.
        Streamlit's st.write_stream() consumes this generator directly.

        Usage:
            for chunk in generator.stream_answer(context, question):
                print(chunk, end="", flush=True)
        """

        if not context.strip():
            yield "I could not find this information in the provided document."
            return

        # Prevent accidentally sending huge context windows
        context = context[:MAX_CONTEXT_CHARS]

        user_prompt = build_user_prompt(context, question)

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    },
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content

                if delta is not None:
                    yield delta

        except Exception as e:
            print(f"Groq API Error: {e}")
            yield "LLM service temporarily unavailable. Please try again."

    def get_answer(self, context: str, question: str) -> str:
        """
        Non-streaming version. Returns the full answer as a string.
        Useful for testing and scripts.
        """

        if not context.strip():
            return "I could not find this information in the provided document."

        full_answer = ""

        for chunk in self.stream_answer(context, question):
            full_answer += chunk

        return full_answer


# ──────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":

    print("\nGenerator Test\n")

    # Minimal test context (normally comes from retriever)
    test_context = """[Source 1]
Sellers must meet eBay's minimum performance standards.
Failure to meet these standards may result in eBay charging
sellers additional fees, and/or limiting, restricting,
suspending, or downgrading your seller account.
"""

    test_question = (
        "What happens if a seller fails to meet performance standards?"
    )

    gen = Generator()

    print(f"Question: {test_question}\n")
    print("Answer (streaming):\n")

    for token in gen.stream_answer(test_context, test_question):
        print(token, end="", flush=True)

    print("\n\nGenerator test complete.\n")