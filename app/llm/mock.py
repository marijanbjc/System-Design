"""Deterministic mock LLM. Default provider so the demo runs without an API key.

It does not paraphrase: it stitches the retrieved chunks into an answer and derives
its flags from measurable properties of the input. That keeps the demo reproducible
and makes the gate observable — the same ticket always takes the same branch.
"""

import re

from app.llm.base import LLMDraft, LLMUnavailable, build_prompt
from app.pii import contains_pii

_OFF_TOPIC_MARKERS = re.compile(
    r"\b(погод\w+|политик\w+|президент|курс\s+валют|рецепт|футбол)\b", re.IGNORECASE
)
_TOXIC_MARKERS = re.compile(r"\b(идиот|дебил|тварь|ублюд\w+|убью)\b", re.IGNORECASE)
_FAIL_MARKER = "__LLM_FAIL__"  # lets the demo exercise the degradation path


class MockLLM:
    """Contract-compatible stand-in for a real provider."""

    model_name = "mock-deterministic-v1"

    def generate(self, question_scrubbed: str, context_chunks: list[str]) -> tuple[LLMDraft, int]:
        if _FAIL_MARKER in question_scrubbed:
            raise LLMUnavailable("mock provider forced failure")

        # Belt and braces: the scrubbing stage runs before this call, but if raw PII
        # ever reached a real provider it would be unrecoverable. Fail loudly instead.
        if contains_pii(question_scrubbed):
            raise LLMUnavailable("refusing to process text that still contains PII")

        prompt = build_prompt(question_scrubbed, context_chunks)
        tokens = max(1, len(prompt) // 4)

        context_score = self._context_score(question_scrubbed, context_chunks)
        is_on_topic = not bool(_OFF_TOPIC_MARKERS.search(question_scrubbed))
        is_toxic = bool(_TOXIC_MARKERS.search(question_scrubbed))

        if context_chunks and context_score >= 0.5 and is_on_topic and not is_toxic:
            draft = (
                f"По вашему вопросу: {context_chunks[0].strip()} "
                "Если что-то осталось непонятным, ответьте на это сообщение."
            )
            confidence = min(0.95, 0.55 + context_score * 0.4)
        else:
            draft = "Недостаточно данных для ответа, передаю обращение оператору."
            confidence = 0.25

        return (
            LLMDraft(
                answer_draft=draft,
                has_enough_context=context_score,
                is_on_topic=is_on_topic,
                is_toxic=is_toxic,
                confidence=round(confidence, 3),
            ),
            tokens,
        )

    @staticmethod
    def _context_score(question: str, chunks: list[str]) -> float:
        """Lexical overlap between the question and the retrieved context.

        A stand-in for the model's own judgement of context sufficiency: deterministic,
        and it moves in the right direction when retrieval is poor.
        """
        if not chunks:
            return 0.0
        # Placeholders left by scrubbing carry no meaning; counting them in the
        # denominator would understate context sufficiency for every ticket with PII.
        without_placeholders = re.sub(r"\[[A-Z]+_\d+\]", " ", question)
        question_words = {w for w in re.findall(r"\w{4,}", without_placeholders.lower())}
        if not question_words:
            return 0.0
        chunk_words = {w for w in re.findall(r"\w{4,}", " ".join(chunks).lower())}
        return round(len(question_words & chunk_words) / len(question_words), 3)
