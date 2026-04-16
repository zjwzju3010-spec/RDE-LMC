import sys
import os
import json
import re
import time

# Add project root to path so we can import API.py
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from API import chat as _raw_chat
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MODEL = "gpt-4o"


def call_llm(prompt: str, model: str = None, expect_json: bool = False):
    """
    Call LLM with retry logic. Returns str or dict (if expect_json=True).
    Retries up to 3 times on failure with exponential backoff.
    """
    if model is None:
        model = DEFAULT_MODEL

    last_error = None
    for attempt in range(3):
        try:
            response = _raw_chat(model, prompt)
            if expect_json:
                return extract_json_from_response(response)
            return response
        except (json.JSONDecodeError, ValueError) as e:
            last_error = e
            logger.warning(f"JSON parse failed (attempt {attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)
        except Exception as e:
            last_error = e
            logger.warning(f"LLM call failed (attempt {attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)

    if expect_json:
        raise ValueError(f"Failed to get valid JSON after 3 attempts: {last_error}")
    raise RuntimeError(f"LLM call failed after 3 attempts: {last_error}")


def extract_json_from_response(text: str) -> dict:
    """
    Extracts JSON from LLM response that may be wrapped in markdown code blocks.
    Handles:  ```json {...} ``` and ``` {...} ``` and raw {...}
    """
    if not text:
        raise ValueError("Empty response from LLM")

    # Try to find JSON in code blocks first
    patterns = [
        r'```json\s*([\s\S]*?)\s*```',
        r'```\s*([\s\S]*?)\s*```',
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    # Try raw JSON (find first { to last })
    text = text.strip()
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        try:
            return json.loads(text[first_brace:last_brace+1])
        except json.JSONDecodeError:
            pass

    # Last resort: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Cannot extract JSON from response: {text[:200]}... Error: {e}")
