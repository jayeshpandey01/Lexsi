"""Tool contracts, parameter schemas, and normalized error hierarchies."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ToolParameterSchema(BaseModel):
    """Schema descriptor for a single tool argument."""

    name: str = Field(..., description="Unique parameter identifier")
    type: str = Field(..., description="Primitive type: 'string', 'integer', 'float', 'boolean', 'list', 'dict'")
    description: str = Field(..., description="Functional semantic explanation")
    required: bool = Field(True, description="Whether parameter is mandatory")
    enum: Optional[List[str]] = Field(None, description="Allowed values if categorical")


class ToolContract(BaseModel):
    """Formal specification of an executable agent tool."""

    name: str = Field(..., description="Unique tool name")
    description: str = Field(..., description="Precise description of tool capability")
    parameters: List[ToolParameterSchema]
    is_stateful: bool = Field(False, description="Whether tool mutates environment state")
    idempotent: bool = Field(True, description="Whether repeated calls produce identical results")


class ToolExecutionError(Exception):
    """Base exception for all tool execution anomalies."""

    def __init__(self, message: str, error_code: str) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code

    def to_dict(self) -> Dict[str, str]:
        return {"error": self.message, "error_code": self.error_code}


class ToolNotFoundError(ToolExecutionError):
    """Raised when the agent attempts to invoke an unregistered or unknown tool."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(f"Tool '{tool_name}' is not recognized or authorized.", "TOOL_NOT_FOUND")


class SchemaValidationError(ToolExecutionError):
    """Raised when arguments violate parameter types, required fields, or enum constraints."""

    def __init__(self, details: str) -> None:
        super().__init__(f"Argument schema validation failed: {details}", "SCHEMA_VALIDATION_ERROR")


class DisallowedToolError(ToolExecutionError):
    """Raised when policy guard rejects a tool action due to permission breach."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(f"Tool '{tool_name}' is prohibited by current policy guard.", "DISALLOWED_TOOL")
