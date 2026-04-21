from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

THINKING_PATTERN = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE)
ACTION_TAG_PATTERN = re.compile(r"<action\s+id=[\"']([^\"']+)[\"']([^/]*?)/>", re.DOTALL | re.IGNORECASE)
PARAM_PATTERN = re.compile(r"(\w+)=['\"]([^'\"]*)['\"]", re.DOTALL)
REVERSIBILITY_TAG_PATTERN = re.compile(
    r"<reversibility\s+level=[\"']([Rr][1-5])[\"'](?:\s+confidence=[\"']([^\"']*)[\"'])?\s*/>",
    re.DOTALL | re.IGNORECASE,
)


@dataclass
class ParsedAgentOutput:
    action_id: Optional[str]
    parameters: Dict[str, str]
    predicted_r_level: Optional[int]
    predicted_confidence: Optional[float]
    raw_thinking: Optional[str]
    parse_errors: List[str] = field(default_factory=list)


def _safe_parse_float(value_str: Optional[str]) -> Optional[float]:
    if value_str is None:
        return None

    cleaned = value_str.strip()
    cleaned = re.split(r"[\s(]", cleaned)[0]
    cleaned = cleaned.lstrip("~≈<>")

    try:
        result = float(cleaned)
    except (TypeError, ValueError):
        return None

    return max(0.0, min(1.0, result))


def parse_agent_output(text: str) -> ParsedAgentOutput:
    errors: List[str] = []

    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"```", "", text)

    thinking_match = THINKING_PATTERN.search(text)
    raw_thinking = thinking_match.group(1).strip() if thinking_match else None

    action_match = ACTION_TAG_PATTERN.search(text)
    if not action_match:
        errors.append("No <action id='...' .../> tag found in output")
        return ParsedAgentOutput(
            action_id=None,
            parameters={},
            predicted_r_level=None,
            predicted_confidence=None,
            raw_thinking=raw_thinking,
            parse_errors=errors,
        )

    action_id = action_match.group(1).strip()
    parameter_string = action_match.group(2) or ""

    parameters: Dict[str, str] = {}
    for match in PARAM_PATTERN.finditer(parameter_string):
        key = match.group(1).strip()
        value = match.group(2).strip()
        if key.lower() != "id":
            parameters[key] = value

    rev_match = REVERSIBILITY_TAG_PATTERN.search(text)
    predicted_r_level: Optional[int] = None
    predicted_confidence: Optional[float] = None

    if rev_match:
        level_str = rev_match.group(1).upper()
        confidence_str = rev_match.group(2)

        try:
            level_num = int(level_str[1])
            if 1 <= level_num <= 5:
                predicted_r_level = level_num
            else:
                errors.append(f"R-level {level_num} out of range 1-5")
        except (IndexError, ValueError):
            errors.append(f"Cannot parse R-level from '{level_str}'")

        predicted_confidence = _safe_parse_float(confidence_str)
        if confidence_str and predicted_confidence is None:
            errors.append(
                f"Cannot parse confidence '{confidence_str}' as float - prediction score will be 0 for this step"
            )
    else:
        errors.append("No <reversibility level='...' confidence='...'/> tag found - prediction score will be 0 for this step")

    return ParsedAgentOutput(
        action_id=action_id,
        parameters=parameters,
        predicted_r_level=predicted_r_level,
        predicted_confidence=predicted_confidence,
        raw_thinking=raw_thinking,
        parse_errors=errors,
    )
