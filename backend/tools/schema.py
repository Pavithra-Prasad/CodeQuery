"""
Turn typed Python functions with Google-style docstrings into Ollama tool schemas,
and check/convert the arguments a model sends back before calling the function.
"""

import inspect
import logging
import re
from typing import Any, Callable, get_type_hints

logger = logging.getLogger("codequery.tools")

_JSON_TYPES = {str: "string", int: "integer", float: "number", bool: "boolean"}
_SECTION = re.compile(r"^(Args|Arguments|Returns|Raises|Yields|Examples?):\s*$")
_ARG_LINE = re.compile(r"^(\w+)\s*(?:\([^)]*\))?:\s*(.*)$")


def parse_docstring(doc: str) -> tuple[str, dict[str, str]]:
    """Split a Google-style docstring into its description and a {argument: description} map."""
    description, args = [], {}
    section, current = None, None

    for line in inspect.cleandoc(doc or "").splitlines():
        stripped = line.strip()
        header = _SECTION.match(stripped)
        if header and not line.startswith(" "):
            section, current = header.group(1), None
            continue
        if section is None:
            description.append(stripped)
        elif section in ("Args", "Arguments") and stripped:
            match = _ARG_LINE.match(stripped)
            if match and line.startswith("    ") and not line.startswith("        "):
                current = match.group(1)
                args[current] = match.group(2)
            elif current:
                args[current] = f"{args[current]} {stripped}".strip()

    return " ".join(part for part in description if part), args


def function_to_tool(func: Callable) -> dict:
    """Build an Ollama/OpenAI-style tool schema from a function's signature and docstring."""
    signature = inspect.signature(func)
    hints = get_type_hints(func)
    description, arg_docs = parse_docstring(func.__doc__ or "")

    properties, required = {}, []
    for name, param in signature.parameters.items():
        prop: dict[str, Any] = {"type": _JSON_TYPES.get(hints.get(name, str), "string")}
        if name in arg_docs:
            prop["description"] = arg_docs[name]
        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            prop["default"] = param.default
        properties[name] = prop

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


def _coerce(name: str, value: Any, type_: Any) -> Any:
    """Convert a model-supplied value to the annotated type (models often send numbers as strings)."""
    if type_ is int:
        if isinstance(value, bool):
            raise ValueError(f"'{name}' must be an integer, got {value!r}")
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"'{name}' must be an integer, got {value!r}") from None
        if not number.is_integer():
            raise ValueError(f"'{name}' must be a whole number, got {value!r}")
        return int(number)
    if type_ is float:
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"'{name}' must be a number, got {value!r}") from None
    if type_ is str:
        if isinstance(value, (dict, list)):
            raise ValueError(f"'{name}' must be a string, got {type(value).__name__}")
        return value if isinstance(value, str) else str(value)
    return value


def bind_arguments(func: Callable, arguments: dict) -> dict:
    """
    Check a model's tool arguments against the function signature.

    Unknown arguments are dropped (models sometimes invent extras), values are converted to the
    annotated types, and missing required arguments raise.

    Raises:
        TypeError: a required argument is missing.
        ValueError: a value can't be converted to its type.
    """
    signature = inspect.signature(func)
    hints = get_type_hints(func)

    extra = sorted(set(arguments) - set(signature.parameters))
    if extra:
        logger.warning("Ignoring unknown argument(s) for %s: %s", func.__name__, ", ".join(extra))

    kwargs = {}
    for name, param in signature.parameters.items():
        if name in arguments:
            kwargs[name] = _coerce(name, arguments[name], hints.get(name))
        elif param.default is inspect.Parameter.empty:
            raise TypeError(f"missing required argument '{name}'")
    return kwargs
