# Using an Integrated Application Model in SystemFlow

This document shows how to load an application resource model produced by the
Agentic Scientific Modeling workflow, evaluate it directly, and use it as a
mutation in a SystemFlow `ExecutionGraph`.

## Automatic publication after integration approval

The SystemFlow Integration Agent publishes the approved assets when the final
`systemflow_integration_review` is approved and `./agentic resume` runs. No
manual copy is required. For the current application, SystemFlow receives:

```text
/Users/zhongzheng/Desktop/workspace/systemflow/systemflow/
  application_model_data/pty-chi/
    manifest.json
    workflow_application_resource_model.v002.json
    systemflow_application_mapping.v001.yaml
    systemflow_integration_report.v002.yaml
```

`manifest.json` is the stable entry point. It records the active versioned file
names, their SHA-256 hashes, the runtime module, and whether the measurements
are suitable for scientific use. The current five-row pipeline fixture is
published with `scientific_use: false`.

## Runtime and integration artifacts

The runtime implementation is application-independent:

```text
/Users/zhongzheng/Desktop/workspace/systemflow/systemflow/application_models.py
```

It provides:

- `WorkflowApplicationResourceModel`, which loads and evaluates a reviewed model
  JSON file;
- `ScientificApplicationModel`, which exposes predictions as a SystemFlow
  mutation;
- `ApplicationInputSource`, which supplies inputs mapped from message fields or
  properties.

Before final approval, the current PtyChi integration assets remain workflow
artifacts:

```text
runs/ptychi_001/artifacts/systemflow_mapping/
  workflow_application_resource_model.v002.json

runs/ptychi_001/artifacts/systemflow_mapping_review/
  systemflow_application_mapping.v001.yaml
```

The model JSON contains fitted features, coefficients, target definitions,
supported input ranges, and hardware/algorithm groups. The mapping YAML defines
where SystemFlow reads each input and where it writes each prediction. Fitted
coefficients should remain in the JSON artifact and should not be copied into
`application_models.py`.

## Direct prediction

Use this when a complete `ExecutionGraph` is unnecessary and only the numerical
resource estimate is needed:

```python
from pathlib import Path
import json
import sys

SYSTEMFLOW_ROOT = Path("/Users/zhongzheng/Desktop/workspace/systemflow")
DEPLOYMENT_ROOT = SYSTEMFLOW_ROOT / "systemflow/application_model_data/pty-chi"
manifest = json.loads((DEPLOYMENT_ROOT / "manifest.json").read_text(encoding="utf-8"))
MODEL_PATH = DEPLOYMENT_ROOT / manifest["assets"]["model"]["path"]

sys.path.insert(0, str(SYSTEMFLOW_ROOT))

from systemflow.application_models import WorkflowApplicationResourceModel

model = WorkflowApplicationResourceModel(MODEL_PATH)

estimate = model.predict(
    inputs={
        "scan_point_count": 64,
        "detector_shape": [64, 64],
        "num_epochs": 2,
        "batch_size": 1,
    },
    group_selectors={
        "accelerator": "GH200",
        "algorithm": "pie",
    },
)

print(estimate.predictions)
print(estimate.metadata)
```

The prediction dictionary contains:

```python
{
    "latency_s": float,
    "avg_power_w": float,
    "energy_j": float,
    "peak_memory_mib": float,
    "throughput_per_s": float,
}
```

## Prediction inside an ExecutionGraph

This is the normal integration path. The mapping controls input sources, group
selectors, output destinations, names, and type conversions.

```python
from pathlib import Path
import json
import sys
import yaml

SYSTEMFLOW_ROOT = Path("/Users/zhongzheng/Desktop/workspace/systemflow")
DEPLOYMENT_ROOT = SYSTEMFLOW_ROOT / "systemflow/application_model_data/pty-chi"
manifest = json.loads((DEPLOYMENT_ROOT / "manifest.json").read_text(encoding="utf-8"))
MODEL_PATH = DEPLOYMENT_ROOT / manifest["assets"]["model"]["path"]
MAPPING_PATH = DEPLOYMENT_ROOT / manifest["assets"]["mapping"]["path"]

sys.path.insert(0, str(SYSTEMFLOW_ROOT))

from systemflow.application_models import (
    ApplicationInputSource,
    ScientificApplicationModel,
    WorkflowApplicationResourceModel,
)
from systemflow.node import Component, DefaultLink, ExecutionGraph

resource_model = WorkflowApplicationResourceModel(MODEL_PATH)
mapping = yaml.safe_load(MAPPING_PATH.read_text(encoding="utf-8"))

# The current mapping reads all fitted inputs and group selectors from the
# application component's parameters. Therefore the source has no parameters.
source = Component(
    "Application inputs",
    [ApplicationInputSource(mapping)],
    parameters={},
)

application = Component(
    mapping["component_name"],
    [ScientificApplicationModel(resource_model, mapping)],
    parameters={
        "scan_point_count_patterns": 64,
        "detector_shape_pixels": [64, 64],
        "reconstruction_epoch_count_epochs": 2,
        "minibatch_size_patterns_per_minibatch": 1,
        "accelerator_model": "GH200",
        "algorithm_group": "pie",
    },
)

graph = ExecutionGraph(
    "PtyChi resource prediction",
    nodes=[source, application],
    links=[
        DefaultLink(
            "inputs to application",
            source.name,
            application.name,
        )
    ],
)

result = graph()
```

## Reading integrated predictions

The approved mapping writes latency to the final message:

```python
latency_s = result.root_node.output_msg.properties["application_latency_s"]
```

It writes power, energy, memory, and throughput to the application component:

```python
application_result = result.get_node(mapping["component_name"])
properties = application_result.properties

avg_power_w = properties["application_avg_power_w"]
energy_j = properties["application_energy_j"]
peak_memory_mib = properties["application_peak_memory_mib"]
throughput_per_s = properties["application_throughput_per_s"]
metadata = properties["application_resource_model_metadata"]

print(
    {
        "latency_s": latency_s,
        "avg_power_w": avg_power_w,
        "energy_j": energy_j,
        "peak_memory_mib": peak_memory_mib,
        "throughput_per_s": throughput_per_s,
        "metadata": metadata,
    }
)
```

For the current pipeline fixture, the integration validation produced
approximately:

```text
latency_s          1.311746
avg_power_w       95.924081
energy_j         845.091150
peak_memory_mib   24.0
throughput_per_s  48.789933
```

## Input and group requirements

Input keys passed to `predict()` must exactly match the model contract:

```text
scan_point_count
detector_shape
num_epochs
batch_size
```

Group selectors must exactly match an available fitted group. The current model
supports:

```text
accelerator = GH200
algorithm   = pie
```

The current supported domain is a single smoke point:

```text
scan_point_count = 64
detector_shape   = [64, 64]
num_epochs       = 2
batch_size       = 1
```

Do not treat predictions outside this domain as validated. A production model
should replace this artifact with one fitted from multiple real, distinct input
points spanning the intended operating range.

## Pipeline-fixture limitation

The current resource model was fitted from five pipeline-fixture rows that all
duplicate one real successful GH200 run. It verifies loading, mapping, graph
execution, and output propagation, but it does not measure scaling behavior.
Consequently, its non-intercept coefficients are zero and its predictions must
not be used for scientific inference, capacity planning, or hardware comparison.

When a production experiment is complete, point `MODEL_PATH` at the newly
reviewed `workflow_application_resource_model.vNNN.json`. The generic
`application_models.py` runtime and graph construction code do not need to be
rewritten.
