# Agentic AI Workflow for Scientific Application Resource Modeling

## Purpose

This workflow builds system-level performance and energy models for a scientific application as a function of its inputs.

The primary objective is not to compare applications or backends. Given one scientific application, the workflow should determine:

- which application inputs affect the workload
- how those inputs affect FLOPs, data volume, memory use, and execution phases
- which benchmark points are needed to characterize the application
- how latency, power, and energy should be measured
- how measured data should be validated
- how parameterized resource models should be fitted
- how the resulting models should be integrated into SystemFlow

The target model has the general form:

```text
(application inputs, execution configuration, hardware)
    -> (latency, power, energy, throughput, memory)
```

For example:

```text
(number of images, resolution, batch size, iterations, GPU)
    -> (end-to-end latency, total energy)
```

The workflow automates application analysis, experiment design, benchmark execution, data validation, model fitting, and SystemFlow integration. Human researchers remain responsible for confirming application semantics, approving measurement boundaries, accepting assumptions, authorizing expensive runs, and approving final models.

## Core Modeling Principles

### Model one application across an input space

The central object is a scientific application with a parameterized workload.
Alternative algorithms are separate model groups that share one approved input
vector. Algorithm identity is not a model input; experiments reuse the same
input points for each group, and resource modeling fits separate coefficients.

### Separate theoretical work from measured resource use

The workflow distinguishes:

- **algorithmic FLOPs**: useful operations implied by the algorithm
- **executed FLOPs**: operations actually executed, including padding or other implementation effects
- **achieved FLOP/s**: hardware-specific performance measured at runtime
- **data movement**: input, output, host-device, and memory traffic
- **latency and energy**: system-level quantities measured over an explicitly defined boundary

FLOPs alone are not sufficient to predict performance or energy. I/O volume, memory behavior, initialization, communication, and hardware utilization may be equally important.

### Make the system boundary explicit

Every plan must define what is included in the measurement. A system-level measurement may include:

```text
application startup
    -> input loading
    -> preprocessing
    -> host-to-device transfer
    -> core computation
    -> device-to-host transfer
    -> postprocessing
    -> output writing
```

End-to-end and phase-level measurements should be kept distinct.

### Keep scientific logic auditable

LLMs can help inspect source code, interpret configuration, identify likely inputs, and draft formulas. Deterministic code should validate schemas, evaluate formulas, expand experiment matrices, compute run counts, and enforce approval rules. Every inferred input or workload equation should include evidence, assumptions, and a confidence level.

## Five-Agent Workflow

```text
Application Characterization Agent
    -> discover candidate inputs
    -> Human Gate 1: approve/edit/reject candidate inputs
    -> derive theoretical compute and I/O model from approved inputs
    -> Human Gate 2: approve/edit/reject complete characterization

Experiment Planning Agent
    -> human approval of prediction domain, measurement protocol, and run matrix

Synthetic Dataset Preparation
    -> deterministic, validated application-compatible inputs and manifest

Benchmark Runner Agent
    -> raw logs, power traces, profiler outputs, and run manifest

Data Validation & Resource Modeling Agent
    -> human approval of usable measurements, model form, and fitted coefficients

SystemFlow Integration & Report Agent
    -> human approval of the integrated model and final report
```

The workflow may be iterative. If resource modeling finds poor coverage or
large uncertainty, it can request additional benchmark points from the
Experiment Planning Agent.

## Orchestration and Artifact Contract

The five responsibilities are implemented as stages in one persistent
LangGraph workflow. Agents do not freely invoke one another. Graph edges,
validated artifacts, and explicit human decisions control every transition.

- SQLite checkpoints make interrupts and failed nodes resumable across processes.
- The graph state stores small references, not full scientific artifacts.
- Artifacts are immutable, numbered YAML files with SHA-256 references.
- Human review is submitted as YAML with `approve`, `edit`, or `reject`.
- An edit is ingested as a new artifact version; an existing version is never overwritten.
- Deterministic nodes validate schemas, hashes, stale reviews, and routing.
- LLM nodes inspect code and draft scientific content but do not approve their own output.

## 1. Application Characterization Agent

### Responsibility

The first agent analyzes a scientific application before any expensive
benchmarking begins. It discovers meaningful inputs, traces how they affect
computation and data movement, and constructs an evidence-backed symbolic
workload model. It does not select benchmark points or execute the application.

### Inputs

- application source repository or package
- application entry point, CLI, API, notebook, or job script
- example invocation and representative input
- configuration files
- optional publications or algorithm descriptions

### Step 1: Discover Application Inputs

The agent should inspect:

- command-line arguments
- configuration objects and files
- public APIs
- dataset loaders
- tensor or array shapes
- model and solver configuration
- loop bounds and convergence conditions
- example scripts, notebooks, and documentation

It should classify discovered parameters:

| Class | Examples |
|---|---|
| Scientific input | number of images, events, particles, grid size |
| Problem shape | resolution, dimensions, channels |
| Algorithm parameter | iterations, epochs, tolerance, solver |
| Execution parameter | batch size, worker count, precision |
| Hardware parameter | GPU, CPU count, memory |
| Reproducibility parameter | random seed |
| Operational parameter | output directory, logging verbosity |

Not every configuration field belongs in the resource model. The agent should explain why each selected variable is or is not a modeling input.

Each discovered input should contain provenance:

```yaml
- name: resolution
  symbol: N
  type: integer
  units: pixels
  role: problem_shape
  affects: [input_bytes, flops, memory]
  evidence:
    file: application/config.py
    line: 42
  confidence: high
```

### Human Gate 1: Confirm Candidate Inputs

The graph pauses after input discovery. The researcher confirms which inputs
belong in the workload model and future experiment sweep, and may approve,
edit, or reject the candidate-input artifact. Formula derivation starts only
from the approved input version.

### Step 2: Identify Execution Phases

The agent should identify important system-level phases and their boundaries:

- initialization and model loading
- input I/O
- preprocessing
- host-device transfer
- core compute
- communication or synchronization
- device-host transfer
- postprocessing
- output I/O

It should state which phases are included in end-to-end latency, end-to-end energy, and phase-level measurements. If phase boundaries are not observable, the agent should recommend instrumentation points rather than silently inventing them.

### Step 3: Construct a Parameterized Workload Model

The workload model should express theoretical work as functions of application inputs. It may include:

- number of samples and batches
- tensor or array shapes
- input and output bytes
- algorithmic and executed FLOPs
- estimated peak memory
- communication volume
- operation counts by phase

Example for batched inference:

```text
K(R,B) = ceil(S(R) / B)
F_useful(R,N) = S(R) * F_sample(N)
F_executed(R,N,B) = K(R,B) * B * F_sample(N)   # if padded batches execute
```

Example for an iterative FFT-based application:

```text
FFT2_flops(N) = 10 * N^2 * log2(N)
F_image_iteration(N) = q_fft * FFT2_flops(N) + q_elementwise * N^2
F_total(R,N,I) = R * I * F_image_iteration(N)
```

The formulas must declare assumptions. When a quantity cannot be derived statically, the agent should mark it for profiler or benchmark validation:

```yaml
total_flops:
  expression: "R * I * (q_fft * 10 * N**2 * log2(N))"
  method: analytical
  assumptions:
    - square two-dimensional input
    - fixed iteration count
  evidence:
    - file: application/reconstruction.py
      operation: fft2
  confidence: medium
  validation_required: true
```

Evidence may come from source-level operator and shape analysis, framework FLOP counters, profiler output, algorithm documentation, or publications. Estimates based only on analogy or LLM inference must be labeled low-confidence.

### Outputs

- versioned `candidate_inputs.yaml` artifacts and review records
- versioned `application_characterization.yaml` artifacts and review records
- provenance containing source revision, prompt/agent versions, and SHA-256

### Human Gate 2: Approve Complete Characterization

After candidate inputs have already been accepted, the researcher confirms
formula dependencies, execution phases, system boundary, synthetic-input
requirements, assumptions, and unresolved work. Only an explicitly approved
characterization artifact reference may enter planning.

## 2. Experiment Planning Agent

### Responsibility

The Planning Agent converts one approved characterization into a reviewable
experiment contract. It selects a prediction domain, pilot sample points,
hardware subset, repetitions, synthetic-input specification, and measurement
protocol. It produces a deterministic dry-run matrix but does not run it.
When the application provides multiple algorithms, the plan applies the same
input-space design to every selected algorithm group. Later modeling estimates
independent coefficient sets rather than treating algorithm as a categorical
input feature.

### Inputs

- approved `application_characterization.yaml`
- optional research objective and intended prediction domain
- allowed accelerator catalog and selected target subset
- measurement requirements and run budget
- application source through read-only tools when planning details are missing

### Step 1: Define the Modeling Objective

The plan must identify dependent and independent variables:

```yaml
modeling_objective:
  predict:
    - end_to_end_latency_s
    - total_energy_j
    - throughput_samples_per_s
  as_function_of:
    - number_of_images
    - resolution
    - batch_size
    - iterations
    - hardware_id
  intended_prediction_domain:
    number_of_images: [1000, 16000]
    resolution: [64, 256]
```

The intended prediction domain matters because a fitted model should not be silently used far outside the measured range.

### Step 2: Design the Benchmark Matrix

The agent selects benchmark points that provide enough information to calibrate workload and resource models without requiring an unnecessarily large Cartesian product.

It should consider:

- linear, logarithmic, or domain-specific sampling
- boundary points
- pilot versus full experiments
- one-factor and interaction coverage
- repetitions and random seeds
- warm-up runs
- invalid or unsupported input combinations
- resource budgets and maximum run count
- expected runtime and storage

Example:

```yaml
sampling:
  number_of_images:
    strategy: logarithmic
    values: [1000, 2000, 4000, 8000, 16000]
  resolution:
    strategy: categorical
    values: [64, 128, 256]
  batch_size:
    strategy: selected
    values: [256, 512, 1024]
  iterations:
    strategy: selected
    values: [1, 2, 4]
```

The agent must calculate the resulting run count deterministically and may recommend a smaller pilot matrix when the full design exceeds the budget.

The deterministic measured-run formula is:

```text
algorithm groups × shared base points × hardware targets × repetitions
```

Algorithm identity is a grouping column in the matrix, not an input variable.

### Step 3: Define the Measurement Protocol

The plan should specify:

- warm-up policy and measured repetitions
- timer and synchronization requirements
- power measurement target: accelerator, CPU, or whole node
- power sampling interval
- clock alignment between logs and power traces
- profiler requirements
- system metadata to record
- expected output artifacts

For short runs, the agent should verify that the power sampling frequency is sufficient for meaningful energy integration.

### Validation and Feasibility Checks

The agent should flag:

- missing or ambiguous modeling inputs
- unbounded or runtime-dependent loop counts
- insufficient sampling of a proposed model variable
- a requested energy model without a power measurement method
- input points that exceed device memory
- unsupported input or hardware combinations
- runs too short for the selected power sampling interval
- experiments that vary too many coupled parameters without enough coverage
- plans that exceed run-count, wall-time, or storage budgets
- inconsistent system boundaries across runs
- formulas without evidence or assumptions
- extrapolation beyond the benchmark domain

Validation severity should be explicit:

```text
ERROR   prevents approval or execution
WARNING requires human review
INFO    records assumptions or recommendations
```

### Outputs

- versioned `experiment_plan.yaml`
- versioned deterministic `dry_run_matrix.csv`
- archived review submissions and accepted decision records

The plan records the measurement contract. The matrix contains one row per
proposed run and is expanded deterministically from algorithm groups, shared
base points, hardware, and repetitions.

### Human Gate 3: Approve Experiment Plan

Before benchmark execution, the researcher approves:

- the intended prediction domain
- benchmark sample points and repetitions
- target hardware and measurement tools
- synthetic-input specification
- system and measurement boundaries
- estimated experiment cost

The approval state should be stored in the plan. Any material plan change after approval should invalidate the approval and require review again.

## 2.5 Synthetic Dataset Preparation

Dataset preparation is a deterministic execution stage between planning and the
Benchmark Runner. It is not another decision-making agent. It consumes only an
approved experiment plan and generates application-compatible inputs described
by that plan.

For pty-chi, dataset identity is determined by the scientific data dimensions
`(scan_point_count, detector_shape)`. `num_epochs`, `batch_size`, hardware, and
algorithm group affect execution but do not change the input HDF5 contents.
Therefore all matching matrix rows reuse one dataset instead of copying it for
each benchmark run.

The generator must:

- support a no-write preview with unique dataset count and logical storage
- use a recorded deterministic seed for each unique dataset
- stream generation in bounded batches rather than hold diffraction data in memory
- write the exact HDF5 keys, shapes, dtypes, and metadata expected by the loader
- validate finite/nonnegative intensity, probe power, positions, shape, and dtype
- record SHA-256 checksums and generation parameters in per-dataset manifests
- reuse an existing dataset only when its specification and checksums match
- publish a versioned `synthetic_dataset_manifest.yaml` artifact

Dataset preparation must refuse real generation while the experiment plan is
unapproved. The Benchmark Runner consumes the approved plan, deterministic
matrix, and generated dataset manifest together.

## 3. Benchmark Runner Agent

### Responsibility

The Benchmark Runner Agent executes the approved run matrix and collects raw measurements without reinterpreting the research objective. It should support local and HPC execution. Early implementations should default to dry-run mode and generate commands or job scripts for review.

### Inputs

- approved `experiment_plan.yaml`
- `dry_run_matrix.csv`
- `synthetic_dataset_manifest.yaml`
- application entry point and environment specification
- instrumentation configuration
- hardware or scheduler configuration

### Outputs

- stdout and stderr logs
- phase timing logs
- accelerator, CPU, or node power traces
- optional profiler outputs
- environment and hardware metadata
- run manifest
- failed and skipped run lists

The implemented runner first emits an immutable command CSV and run manifest,
then pauses for review. Execution is a separate explicit command. Every run is
identified by a command SHA-256 and writes an individual log, power trace, and
result record. Checkpoint retry reuses matching completed or failed results; it
does not silently rerun failed work.

Every run should record its deterministic run ID, exact inputs, exact invocation, source revision, software environment, hardware identity, timestamps, exit code, and output paths.

### Safety Rules

- Do not execute a plan that is not approved.
- Default to dry-run mode.
- Never delete previous results automatically.
- Skip complete existing runs unless rerun is explicitly approved.
- Do not invent or modify benchmark points.
- Keep stdout, stderr, commands, timestamps, and exit codes.
- Record failures without stopping unrelated runs when safe.

### Human Approval Gate

Approval is required before consuming compute resources. Additional approval may be required for rerunning failed or suspicious measurements.

## 4. Data Validation & Resource Modeling Agent

### Responsibility

This agent converts logs, power traces, profiler results, and manifests into an
auditable dataset, validates measurements, and fits the approved system-level
resource models.

### Parsing Targets

For each run, extract:

- all planned input and configuration values
- phase and end-to-end latency
- average, minimum, maximum, and time-series power
- integrated energy
- profiler operation counts and FLOPs when available
- peak memory
- I/O and transfer volume
- run status and measurement completeness

### Validation Checks

The agent should flag:

- missing, failed, incomplete, or duplicate runs
- observed parameters that differ from the plan
- NaN or negative durations
- clock misalignment between logs and power traces
- insufficient power samples
- power duration inconsistent with application timing
- outliers across repetitions
- impossible power, energy, FLOP, or memory values
- disagreement between analytical and profiler FLOPs
- unexpected hardware or software changes
- measurements outside the approved input domain

### Outputs

- `modeling_runs_extracted.csv`
- `validation_summary.md`
- `missing_runs.csv`
- `failed_runs.csv`
- `suspicious_runs.csv`
- `approved_modeling_dataset.csv`

The implemented validator integrates `GPU*_Power(W)` over monotonic `Time(S)`
samples using the trapezoidal rule and extracts the maximum
`GPU*_MemoryUsed(MiB)`. Missing telemetry makes a run non-usable by default.

### Human Approval Gate

The researcher decides which suspicious runs to retain, exclude, or rerun. Final model fitting must use the explicitly approved dataset.

### Resource Modeling Stage

### Responsibility

The modeling stage fits parameterized system-level models using the approved
benchmark dataset and workload equations produced by the Characterization
Agent.

Possible model structure:

```text
T_total(x,h) = T_startup(h) + T_io(x,h) + T_compute(x,h) + T_output(x,h)
T_compute(x,h) = intercept(h) + F(x) / P_effective(x,h)
E_total(x,h) = integral(P_system(t), t)
```

Here `x` is the application input vector, `h` is the hardware configuration, and `F(x)` is the parameterized workload model.

The agent should not assume that FLOPs alone explain runtime. It should consider input/output bytes, batch count, memory footprint, communication, and execution phases where supported by measurements.

### Responsibilities

- select candidate model forms consistent with application characterization
- fit latency, power, energy, throughput, and memory models as requested
- quantify residuals and uncertainty
- compare analytical FLOPs with profiler measurements
- detect under-sampled regions and parameter interactions
- validate models on held-out benchmark points
- define the supported prediction domain
- recommend additional experiments where fit quality is insufficient

### Outputs

- fitted coefficient tables
- machine-readable model definitions
- fit-quality metrics
- residual and uncertainty plots
- model assumptions summary
- supported-domain metadata
- recommendations for additional benchmark points

The first implemented model form is a standardized ridge model fitted
independently for each `(hardware, algorithm)` group. It predicts end-to-end
latency, average accelerator power, accelerator energy, and peak accelerator
memory from derived features of the four approved inputs. Algorithm identity
remains a grouping key and is never a model input.

### Feedback Loop

If the data cannot identify the proposed model, or error is concentrated in part of the input domain, the agent should request a targeted plan extension:

```text
Data Validation & Resource Modeling Agent
    -> requested additional input points
    -> Experiment Planning Agent
    -> human approval
    -> Benchmark Runner Agent
```

### Human Approval Gate

The researcher approves model form, included measurements, workload assumptions, treatment of warm-up and outliers, uncertainty, supported range, and coefficients allowed to enter SystemFlow.

## 5. SystemFlow Integration & Report Agent

### Responsibility

The SystemFlow Integration & Report Agent packages an approved application resource model for use by SystemFlow, validates predictions through SystemFlow mutations and graphs, and produces a final scientific report.

### Integration Requirements

The integrated model should:

- accept approved application inputs
- evaluate derived workload quantities such as FLOPs and bytes
- select hardware-specific fitted coefficients
- return positive, unit-consistent latency, energy, power, and throughput
- reject or warn about inputs outside the supported domain
- preserve coefficient provenance and model version

Workflow-generated models use the application-independent
`systemflow.application_models.WorkflowApplicationResourceModel`. A mapping
agent inspects the approved application artifacts and drafts a declarative
contract connecting fitted model inputs to `message.fields`,
`message.properties`, or `component.parameters`. It also maps fitted targets
to message or host outputs. The mapping passes a separate human gate before
execution.

Deterministic validation interprets the approved mapping with
`ScientificApplicationModel` and executes actual two-node SystemFlow
`ExecutionGraph` simulations for every fitted model group and planned base
point. No XRS-specific class is required. Domain adapters remain optional
consumers for applications that already have them.

For XRS applications, existing integration patterns may still include:

```python
from systemflow.xrs_models import PtyChiResourceModel, PtychoPINNResourceModel
from systemflow.xrs import PhaseReconstruction2D_model, PhaseReconstruction3D_model
```

These are optional examples, not part of the generic workflow contract and not
a requirement that the analyzed application belongs to XRS.

### Validation Tasks

- load coefficient and model-definition files
- reproduce selected benchmark predictions
- validate units and input mappings
- verify supported hardware and input domains
- check positive latency and energy
- execute representative SystemFlow graphs
- compare SystemFlow outputs with held-out measurements
- ensure out-of-domain inputs produce a warning or error

### Report Contents

- application inputs and intended prediction domain
- system and measurement boundaries
- execution phase decomposition
- analytical FLOP, data-volume, and memory formulas
- benchmark matrix and completeness
- measurement methodology
- fitted performance and energy models
- coefficient tables
- residuals, uncertainty, and validation results
- SystemFlow integration results
- limitations and recommended follow-up experiments

### Human Approval Gate

The researcher approves the final report, model package, and whether the SystemFlow update should be committed or published.

## Human-in-the-Loop Summary

| Stage | Human decision |
|---|---|
| After characterization | Confirm inputs, formulas, phases, and assumptions |
| After planning | Confirm prediction domain, sampling, hardware, measurement, and cost |
| Before execution or reruns | Approve compute and measurement resource use |
| After validation and modeling | Approve usable data, model form, coefficients, uncertainty, and domain |
| After integration | Approve the SystemFlow model and final report |

## Minimal Viable Implementation

The first version should be script-driven rather than fully autonomous:

```text
agentic/
  agentic_ai_workflow.md
  agents/
    characterization/
      agent_design.md
      analysis_request_schema.yaml
      application_characterization_schema.yaml
      prompts/
      runner.py
      tools.py
      cli.py
    planner/
      planner_design.md
      experiment_plan_schema.yaml
      prompts/
      runner.py
      cli.py
    runner/
    generate_run_scripts.py
    run_experiments.py
    validation_modeling/
      parse_measurements.py
      validate_runs.py
      fit_resource_models.py
    integration/
      validate_systemflow.py
      generate_report.py
```

The AI agents should orchestrate deterministic tools, inspect outputs, identify uncertainties, and request approval at defined gates. Scientific logic and workload equations should remain in version-controlled files, not hidden inside prompts.

## First Implementation Milestone

The first milestone implements the first two agents for one selected scientific
application:

1. discover and classify application inputs
2. identify major execution phases
3. construct a parameterized FLOP and data-volume model
4. attach evidence, assumptions, and confidence to each formula
5. consume only an approved characterization
6. define the target prediction domain and measurement protocol
7. propose a pilot matrix and calculate its run count deterministically
8. generate a planning report for human approval

## One-Sentence Description

This is a five-agent, human-in-the-loop workflow that analyzes a scientific application, characterizes how its inputs determine computational work, measures system-level resource use, fits performance and energy models, and integrates the approved models into SystemFlow.
