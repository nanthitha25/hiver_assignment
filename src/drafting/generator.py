"""Grounded Reply Generator using RAG context, LLM, and safety guardrails."""

import logging
from typing import Optional, Tuple
from src.models import RetrievalResult
from src.config import GEMINI_API_KEY, GEMINI_MODEL_NAME, LLM_PROVIDER
from src.drafting.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from src.drafting.guardrails import OutputGuardrail

logger = logging.getLogger(__name__)


class GroundedReplyGenerator:
    """Generates customer support replies grounded in historical brand resolutions."""

    def __init__(
        self,
        provider: str = LLM_PROVIDER,
        api_key: str = GEMINI_API_KEY,
        model_name: str = GEMINI_MODEL_NAME,
    ):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        self.guardrail = OutputGuardrail()
        self._llm = None
        self._init_llm()

    def _init_llm(self):
        """Initializes Gemini API client if API key is provided."""
        if self.provider == "gemini" and self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._llm = genai.GenerativeModel(
                    model_name=self.model_name,
                    system_instruction=SYSTEM_PROMPT.format(retrieved_context="")
                )
                logger.info(f"Initialized Gemini model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}. Falling back to template mode.")
                self._llm = None

    def generate(
        self,
        tweet: str,
        intent: str,
        retrieval_result: Optional[RetrievalResult] = None,
    ) -> Tuple[Optional[str], bool, list]:
        """Generates a grounded reply and returns (reply_text, passed_guardrails, violations)."""
        retrieved_snippets = retrieval_result.snippets if retrieval_result else []
        context_str = "\n".join([f"- {s}" for s in retrieved_snippets]) if retrieved_snippets else "No specific history."

        generated_text = None

        # Try LLM Generation if configured
        if self._llm:
            try:
                system_with_context = SYSTEM_PROMPT.format(retrieved_context=context_str)
                user_prompt = USER_PROMPT_TEMPLATE.format(customer_tweet=tweet, intent=intent)
                full_prompt = f"{system_with_context}\n\n{user_prompt}"

                response = self._llm.generate_content(
                    full_prompt,
                    generation_config={"temperature": 0.2, "max_output_tokens": 120}
                )
                if response and response.text:
                    generated_text = response.text.strip().strip('"')
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}. Using retrieved historical template.")

        # Fallback to top retrieved historical resolution snippet if LLM not available or failed
        if not generated_text:
            if retrieved_snippets:
                generated_text = retrieved_snippets[0]
            else:
                generated_text = "We'd like to help. Have you tried restarting your device? Let us know which iOS version you have."

        # Evaluate against guardrails
        passed, violations = self.guardrail.evaluate(generated_text)

        # Truncate if slight length overflow
        if not passed and any("LENGTH_EXCEEDED" in v for v in violations):
            if len(generated_text) > 280:
                generated_text = generated_text[:277] + "..."
                passed, violations = self.guardrail.evaluate(generated_text)

        return generated_text, passed, violations
