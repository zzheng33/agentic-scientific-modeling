"""Generic application-resource runtime for legacy SystemFlow checkouts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from systemflow.auxtypes import VarCollection
from systemflow.node import Mutate, MutationInputs, MutationOutputs


def _variables(specifications: list[dict[str, Any]]) -> VarCollection:
    return VarCollection(
        **{f"value_{index}": item["key"] for index, item in enumerate(specifications)}
    )


def _transform(value: Any, name: str) -> Any:
    if name == "identity":
        return value
    if name == "int":
        return int(value)
    if name == "float":
        return float(value)
    if name == "str":
        return str(value)
    if name == "tuple_int":
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError("tuple_int requires a two-element list or tuple")
        return tuple(int(item) for item in value)
    if name == "list_int":
        if not isinstance(value, (list, tuple)):
            raise ValueError("list_int requires a list or tuple")
        return [int(item) for item in value]
    raise ValueError(f"Unsupported SystemFlow mapping transform: {name}")


class WorkflowApplicationResourceModel:
    """Load and evaluate an agent-generated application resource model."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve(strict=True)
        self.document = json.loads(self.path.read_text(encoding="utf-8"))
        self.model_inputs = list(self.document["model_inputs"])
        self.grouping = list(self.document["grouping"])
        self.groups = list(self.document["groups"])

    @staticmethod
    def _features(values: dict[str, Any]) -> dict[str, float]:
        scan_points = float(values["scan_point_count"])
        detector_height, detector_width = (
            float(item) for item in values["detector_shape"]
        )
        epochs = float(values["num_epochs"])
        batch_size = min(scan_points, float(values["batch_size"]))
        pixels = detector_height * detector_width
        return {
            "input_gpixels": scan_points * pixels / 1e9,
            "compute_tpixel_epochs": scan_points * pixels * epochs / 1e12,
            "epoch_batches_k": epochs * math.ceil(scan_points / batch_size) / 1e3,
            "batch_mpixels": batch_size * pixels / 1e6,
        }

    def _validate_domain(self, values: dict[str, Any]) -> None:
        height, width = values["detector_shape"]
        flattened = {
            "scan_point_count": values["scan_point_count"],
            "detector_height": height,
            "detector_width": width,
            "num_epochs": values["num_epochs"],
            "batch_size": values["batch_size"],
        }
        for name, value in flattened.items():
            lower, upper = self.document["supported_domain"][name]
            if not float(lower) <= float(value) <= float(upper):
                raise ValueError(
                    f"Application model input {name}={value} is outside "
                    f"supported domain [{lower}, {upper}]"
                )

    def predict(
        self, inputs: dict[str, Any], selectors: dict[str, Any]
    ) -> dict[str, float]:
        self._validate_domain(inputs)
        matches = [
            group
            for group in self.groups
            if all(str(group.get(name)) == str(selectors[name]) for name in self.grouping)
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one resource-model group for selectors {selectors}")
        features = self._features(inputs)
        predictions: dict[str, float] = {}
        for target, model in matches[0]["targets"].items():
            value = float(model["intercept"])
            for feature, coefficient in model["standardized_coefficients"].items():
                mean = float(model["feature_means"][feature])
                scale = float(model["feature_scales"][feature])
                value += float(coefficient) * (features[feature] - mean) / scale
            predictions[target] = max(value, float.fromhex("0x1.0p-1022"))
        return predictions


class ApplicationInputSource(Mutate):
    """Create mapped message inputs for a generic application component."""

    def __init__(self, mapping: dict[str, Any]) -> None:
        self.specifications = [
            specification
            for collection in ("model_input_mapping", "group_mapping")
            for specification in mapping[collection].values()
            if specification["source"].startswith("message.")
        ]
        outputs = MutationOutputs(
            _variables(
                [item for item in self.specifications if item["source"] == "message.fields"]
            ),
            _variables(
                [
                    item
                    for item in self.specifications
                    if item["source"] == "message.properties"
                ]
            ),
            VarCollection(),
        )
        super().__init__(
            "Generic application input source",
            MutationInputs(VarCollection(), VarCollection(), _variables(self.specifications)),
            outputs,
        )

    def transform(self, message, component):
        fields: dict[str, Any] = {}
        properties: dict[str, Any] = {}
        for specification in self.specifications:
            destination = (
                fields if specification["source"] == "message.fields" else properties
            )
            destination[specification["key"]] = component.parameters[specification["key"]]
        return fields, properties, {}


class ScientificApplicationModel(Mutate):
    """Evaluate a fitted model and publish its mapped resource predictions."""

    def __init__(
        self,
        resource_model: WorkflowApplicationResourceModel,
        mapping: dict[str, Any],
    ) -> None:
        self.resource_model = resource_model
        self.mapping = mapping
        all_specs = [
            specification
            for collection in ("model_input_mapping", "group_mapping")
            for specification in mapping[collection].values()
        ]
        inputs = MutationInputs(
            _variables([item for item in all_specs if item["source"] == "message.fields"]),
            _variables(
                [item for item in all_specs if item["source"] == "message.properties"]
            ),
            _variables(
                [item for item in all_specs if item["source"] == "component.parameters"]
            ),
        )
        outputs = MutationOutputs(
            _variables(
                [
                    item
                    for item in mapping["output_mapping"].values()
                    if item["destination"] == "message.fields"
                ]
            ),
            _variables(
                [
                    item
                    for item in mapping["output_mapping"].values()
                    if item["destination"] == "message.properties"
                ]
            ),
            _variables(
                [
                    item
                    for item in mapping["output_mapping"].values()
                    if item["destination"] == "host.properties"
                ]
            ),
        )
        super().__init__("Scientific application resource model", inputs, outputs)

    @staticmethod
    def _read(message, component, specification: dict[str, Any]) -> Any:
        source = specification["source"]
        key = specification["key"]
        if source == "component.parameters":
            value = component.parameters[key]
        elif source == "message.fields":
            value = message.fields[key]
        elif source == "message.properties":
            value = message.properties[key]
        else:
            raise ValueError(f"Unsupported SystemFlow mapping source: {source}")
        return _transform(value, specification["transform"])

    def transform(self, message, component):
        inputs = {
            name: self._read(message, component, specification)
            for name, specification in self.mapping["model_input_mapping"].items()
        }
        selectors = {
            name: self._read(message, component, specification)
            for name, specification in self.mapping["group_mapping"].items()
        }
        predicted = self.resource_model.predict(inputs, selectors)
        fields: dict[str, Any] = {}
        properties: dict[str, Any] = {}
        host: dict[str, Any] = {}
        destinations = {
            "message.fields": fields,
            "message.properties": properties,
            "host.properties": host,
        }
        for target, specification in self.mapping["output_mapping"].items():
            destinations[specification["destination"]][specification["key"]] = predicted[
                target
            ]
        host[self.mapping["metadata_output_key"]] = {
            "inputs": inputs,
            "selectors": selectors,
            "model_path": str(self.resource_model.path),
        }
        return fields, properties, host
