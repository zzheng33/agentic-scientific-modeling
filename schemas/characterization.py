"""Typed contract for the first Characterization input-discovery gate."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CandidateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    classification: Literal[
        "scientific_input",
        "problem_shape",
        "algorithm_parameter",
        "execution_parameter",
        "hardware_parameter",
        "reproducibility_parameter",
        "operational_parameter",
    ]
    data_type: str
    units: str | None = None
    valid_domain: dict[str, Any] | None = None
    default_value: Any = None
    model_input: bool
    affects: list[str]
    exclusion_reason: str | None = None
    evidence: list[str | dict[str, Any]]
    assumptions: list[str]
    confidence: Literal["high", "medium", "low", "unknown"]
    human_decision: str | None = None


class CandidateInputsDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["0.1"] = "0.1"
    analysis_id: str = Field(min_length=1)
    application: dict[str, Any]
    entrypoints: list[dict[str, Any]]
    candidate_inputs: list[CandidateInput]
    requested_decisions: list[str | dict[str, Any]]

    @model_validator(mode="after")
    def validate_unique_inputs(self) -> "CandidateInputsDocument":
        identifiers = [item.input_id for item in self.candidate_inputs]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("candidate input_id values must be unique")
        symbols = [item.symbol for item in self.candidate_inputs]
        if len(symbols) != len(set(symbols)):
            raise ValueError("candidate input symbols must be unique")
        return self
