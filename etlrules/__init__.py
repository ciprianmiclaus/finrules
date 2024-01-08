"""Top-level package for ETLrules."""

from .data import RuleData, context
from .engine import RuleEngine
from .exceptions import (
    ColumnAlreadyExistsError, ExpressionSyntaxError, GraphRuntimeError,
    InvalidPlanError, MissingColumnError, SchemaError, UnsupportedTypeError,
)
from .plan import Plan, PlanMode
from .runner import load_plan, run_plan


__author__ = """Ciprian Miclaus"""
__email__ = "ciprianm@gmail.com"
__version__ = "0.3.2"


__all__ = [
    "RuleData", "context",
    "RuleEngine",
    "ColumnAlreadyExistsError", "ExpressionSyntaxError", "GraphRuntimeError",
    "InvalidPlanError", "MissingColumnError", "SchemaError", "UnsupportedTypeError",
    "Plan", "PlanMode",
    "load_plan", "run_plan",
]
