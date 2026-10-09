"""
NLP Analysis Orchestrator module.
Coordinates input validation, prompt assembly, LLM communication,
JSON parsing, and Pydantic schema validation.
"""

import json
import re
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import ValidationError

from src.llm import LLMService, LLMServiceError
from src.schemas import TicketAnalysisResponse


class AnalyzerError(Exception):
    """Custom exception class for NLP analysis workflow errors."""
    pass


class TicketAnalyzer:
    """
    Main analysis engine that encapsulates the NLP processing pipeline:
    Ticket -> Validation -> Prompt Templating -> LLM Call -> JSON Parsing -> Schema Validation
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the analyzer with a configuration YAML file and prompt template.

        :param config_path: Relative or absolute path to the config YAML file.
        """
        self.config_path = Path(config_path)
        self.config = self._load_yaml_config(self.config_path)
        
        # Load prompt template from path specified in config
        prompt_rel_path = self.config.get("prompt", {}).get(
            "template_path", "prompts/ticket_analysis.txt"
        )
        self.prompt_template = self._load_prompt_template(prompt_rel_path)
        
        # Initialize LLM communication service
        llm_config = self.config.get("llm", {})
        self.llm_service = LLMService(llm_config)

    @staticmethod
    def _load_yaml_config(path: Path) -> Dict[str, Any]:
        """Loads and parses the YAML configuration file."""
        if not path.exists():
            raise AnalyzerError(f"Configuration file not found at: {path.resolve()}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                return config or {}
        except Exception as e:
            raise AnalyzerError(f"Failed to read configuration YAML: {str(e)}")

    @staticmethod
    def _load_prompt_template(rel_path: str) -> str:
        """Reads the prompt template from the filesystem."""
        path = Path(rel_path)
        if not path.exists():
            raise AnalyzerError(f"Prompt template file not found at: {path.resolve()}")
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                if "{ticket}" not in content:
                    raise AnalyzerError("Prompt template must contain '{ticket}' placeholder.")
                return content
        except Exception as e:
            raise AnalyzerError(f"Failed to read prompt template: {str(e)}")

    def validate_ticket_input(self, ticket_text: Optional[str]) -> str:
        """
        Validates the raw customer ticket text before processing.

        :param ticket_text: The user-provided ticket string.
        :return: Cleaned ticket string.
        :raises AnalyzerError: If the ticket is empty, too short, or invalid.
        """
        if ticket_text is None:
            raise AnalyzerError("Ticket input cannot be None.")

        cleaned = ticket_text.strip()
        if not cleaned:
            raise AnalyzerError("Ticket input cannot be empty. Please enter customer ticket text.")

        if len(cleaned) < 5:
            raise AnalyzerError("Ticket text is too short. Please provide at least 5 characters.")

        return cleaned

    @staticmethod
    def _clean_json_markdown(raw_text: str) -> str:
        """
        Strips markdown code blocks (e.g., ```json ... ```) if the model wraps its response in them.
        """
        cleaned = raw_text.strip()
        # Regex to strip ```json ... ``` or ``` ... ```
        pattern = r"^```(?:json)?\s*([\s\S]*?)\s*```$"
        match = re.match(pattern, cleaned, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return cleaned

    def analyze(self, ticket_text: str) -> TicketAnalysisResponse:
        """
        Runs the complete end-to-end NLP analysis pipeline on a single ticket.

        :param ticket_text: Raw customer ticket text.
        :return: Validated TicketAnalysisResponse Pydantic model instance.
        """
        # Step 1: Input Validation
        valid_ticket = self.validate_ticket_input(ticket_text)

        # Step 2: Prompt Construction
        formatted_prompt = self.prompt_template.format(ticket=valid_ticket)

        # Step 3: LLM API Call
        raw_llm_response = self.llm_service.generate(formatted_prompt)

        # Step 4: JSON Parsing
        cleaned_json_str = self._clean_json_markdown(raw_llm_response)
        try:
            parsed_data = json.loads(cleaned_json_str)
        except json.JSONDecodeError as json_err:
            raise AnalyzerError(
                f"Failed to parse LLM response as JSON. Error: {json_err.msg}. "
                f"Raw output preview: {cleaned_json_str[:150]}"
            )

        if not isinstance(parsed_data, dict):
            raise AnalyzerError("LLM response did not parse into a valid JSON object/dictionary.")

        # Step 5: Pydantic Schema Validation
        try:
            validated_response = TicketAnalysisResponse.model_validate(parsed_data)
            return validated_response
        except ValidationError as val_err:
            # Extract readable field-level error messages
            error_details = []
            for err in val_err.errors():
                loc = ".".join(str(l) for l in err.get("loc", []))
                msg = err.get("msg", "invalid")
                error_details.append(f"Field '{loc}': {msg}")
            combined_errors = "; ".join(error_details)
            raise AnalyzerError(f"Response failed schema validation: {combined_errors}")
