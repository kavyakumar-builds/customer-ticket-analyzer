"""
Google Gemini API Client Wrapper module.
Handles initialization, API communication, and resilient error handling.
Supports the modern Google GenAI SDK (google-genai) with fallback support
for google-generativeai.
"""

import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file if available
load_dotenv()


class LLMServiceError(Exception):
    """Custom exception class for sanitized, user-friendly LLM errors."""
    pass


class LLMService:
    """
    Manages interaction with the Google Gemini API.
    Decoupled from UI and prompt formatting logic.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Gemini LLM service with settings from config.yaml and environment variables.

        :param config: Dictionary containing LLM settings (model, temperature, max_tokens, etc.)
        """
        self.config = config
        self.model = config.get("model", "gemini-2.5-flash")
        self.temperature = float(config.get("temperature", 0.1))
        self.max_tokens = int(config.get("max_tokens", 600))
        self.timeout = float(config.get("timeout_seconds", 30))

        # Retrieve API key from environment variables (never hardcoded)
        self.api_key = (
            os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("LLM_API_KEY")
        )

        if not self.api_key or self.api_key.strip() in ("", "your_gemini_api_key_here"):
            raise LLMServiceError(
                "Missing Google Gemini API Key! Please set the 'GEMINI_API_KEY' variable in your '.env' file.\n"
                "Get a free API key at: https://aistudio.google.com/app/apikey"
            )

        self._init_client()

    def _init_client(self):
        """
        Initializes the Gemini client using either the official 'google-genai' SDK
        or legacy 'google-generativeai' SDK if available.
        """
        self._sdk_mode = None

        # Mode 1: Modern official google-genai SDK (recommended)
        try:
            from google import genai
            from google.genai import types, errors
            self.genai = genai
            self.genai_types = types
            self.genai_errors = errors
            self.client = genai.Client(api_key=self.api_key)
            self._sdk_mode = "genai"
            return
        except ImportError:
            pass

        # Mode 2: Classic google-generativeai SDK (legacy fallback)
        try:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=self.api_key)
            self.legacy_genai = legacy_genai
            self._sdk_mode = "legacy"
            return
        except ImportError:
            pass

        raise LLMServiceError(
            "Neither 'google-genai' nor 'google-generativeai' package is installed.\n"
            "Please run: pip install google-genai"
        )

    def generate(self, prompt: str) -> str:
        """
        Sends the formatted prompt to Google Gemini and returns the raw string response.

        :param prompt: The complete prompt string including the ticket content.
        :return: Raw text response from the model.
        :raises LLMServiceError: Sanitized error description if the request fails.
        """
        if not prompt or not prompt.strip():
            raise LLMServiceError("Prompt cannot be empty.")

        if self._sdk_mode == "genai":
            return self._generate_with_genai(prompt)
        elif self._sdk_mode == "legacy":
            return self._generate_with_legacy(prompt)
        else:
            raise LLMServiceError("No valid Google Gemini SDK available.")

    def _generate_with_genai(self, prompt: str) -> str:
        """Invokes Gemini using the modern google-genai SDK with strict JSON response configuration."""
        try:
            config = self.genai_types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
                response_mime_type="application/json",
            )
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )

            if not response or not response.text:
                raise LLMServiceError("The Gemini model returned an empty response or output was filtered.")

            return response.text.strip()

        except self.genai_errors.APIError as e:
            code = getattr(e, "code", None)
            msg = getattr(e, "message", str(e))
            if code in (401, 403) or "API_KEY_INVALID" in msg or "permission" in msg.lower():
                raise LLMServiceError(
                    f"Authentication failed: Invalid Google Gemini API key. Please check GEMINI_API_KEY in your .env file."
                )
            elif code == 429 or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                raise LLMServiceError(
                    "Rate limit exceeded (429): You have reached your Gemini API quota or rate limit. Please try again shortly."
                )
            else:
                raise LLMServiceError(f"Google Gemini API error (Code {code}): {msg}")
        except LLMServiceError:
            raise
        except Exception as e:
            raise LLMServiceError(
                f"Unexpected error communicating with Google Gemini API: {str(e)}"
            )

    def _generate_with_legacy(self, prompt: str) -> str:
        """Invokes Gemini using the legacy google-generativeai SDK."""
        try:
            model = self.legacy_genai.GenerativeModel(
                model_name=self.model,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                    "response_mime_type": "application/json",
                }
            )
            response = model.generate_content(prompt)
            if not response or not response.text:
                raise LLMServiceError("The Gemini model returned an empty response.")
            return response.text.strip()
        except Exception as e:
            err_str = str(e)
            if "API_KEY_INVALID" in err_str or "403" in err_str or "401" in err_str:
                raise LLMServiceError("Authentication failed: Invalid Gemini API key in .env file.")
            elif "429" in err_str or "ResourceExhausted" in err_str or "Quota" in err_str:
                raise LLMServiceError("Rate limit exceeded: Gemini API quota or rate limit exceeded.")
            else:
                raise LLMServiceError(f"Gemini API error: {err_str}")
