# Application Characterization Agent Design

## Goal

The Application Characterization Agent accepts an arbitrary scientific
application codebase at request time and constructs a theoretical description
of how its important inputs affect major compute and I/O work.

The agent does not assume a specific application in advance. It does not run a
profiler, fit hardware performance coefficients, or generate synthetic data in
its first pass. Its first responsibility is to inspect the supplied codebase and
produce an evidence-backed draft for human review.

## Scope

The first implementation analyzes:

- application entry points and configuration paths
- candidate scientific, shape, algorithm, and execution inputs
- major compute phases
- major input, output, and intermediate I/O phases
- parameterized theoretical FLOP formulas
- parameterized I/O byte formulas
- optional memory-capacity formulas when directly derivable
- requirements for creating valid synthetic inputs later

The initial hardware focus is GPU execution on:

- NVIDIA A100, H100, H200, and B200
- AMD MI300A and MI300X
- Intel Data Center GPU Max family

The theoretical workload model should remain hardware-independent wherever
possible. Hardware affects feasibility and later performance/energy calibration,
not the definition of algorithmic FLOPs or application-level I/O bytes.

## Invocation Contract

The application arrives in the user request rather than being registered in the
agent implementation. A minimal request identifies a readable codebase:

```yaml
application_source:
  path: /path/to/application
```

The request may also provide entry-point hints, example commands, documentation,
candidate inputs, or a desired analysis boundary. Missing hints are discoveries
the agent must attempt to make from the repository.

The formal request outline is in `analysis_request_schema.yaml`.

## Two-Pass Human-in-the-Loop Protocol

### Pass 1: Draft characterization

The agent inspects the application and produces:

- discovered entry points
- candidate inputs, including excluded candidates
- input-to-compute dependency formulas
- input-to-I/O dependency formulas
- execution phase boundaries
- synthetic-input requirements
- evidence, assumptions, confidence, and unresolved questions

The draft must have status `awaiting_human_review`. It is not an approved model.

### Human review

The researcher may:

- approve, remove, add, rename, or reclassify inputs
- change which compute or I/O phases are in scope
- correct application semantics
- correct formulas or assumptions
- supply missing documentation or source locations
- mark runtime-dependent quantities that cannot be modeled statically

Human corrections override agent inference and must be recorded as provenance.

### Pass 2: Revised characterization

The agent applies the feedback, rechecks affected dependencies and equations,
and emits a revised artifact. The artifact becomes `approved` only after the
researcher explicitly approves it. Otherwise it returns to
`awaiting_human_review` with remaining questions.

An approved characterization becomes the input to synthetic-input and
experiment planning. Material later changes invalidate approval.

## Analysis Procedure

### 1. Establish repository scope

- read repository instructions before analysis
- inventory languages, build files, packages, and documentation
- identify likely application boundaries
- avoid analyzing vendored code, generated artifacts, or dependencies as if
  they were application logic

### 2. Discover entry points

Inspect CLI definitions, `main` functions, package entry points, workflow files,
job scripts, notebooks, examples, and public APIs. Rank entry points by evidence
and explain ambiguity instead of selecting one silently.

### 3. Trace candidate inputs

Trace parameters from their external definition through configuration and into
shapes, loops, operators, file reads, or file writes. Classify each candidate as:

- `scientific_input`
- `problem_shape`
- `algorithm_parameter`
- `execution_parameter`
- `hardware_parameter`
- `reproducibility_parameter`
- `operational_parameter`

Mark whether it is proposed for the workload model. For excluded parameters,
record a reason.

### 4. Build a dependency graph

Represent important relationships explicitly:

```text
number_of_images
    -> input_bytes
    -> number_of_samples
    -> number_of_batches
    -> total_flops
    -> output_bytes

resolution
    -> elements_per_image
    -> flops_per_image
    -> bytes_per_image
```

Every formula symbol must resolve to a candidate input, a constant with stated
provenance, or another derived quantity.

### 5. Characterize major compute

Identify dominant operators and loops, such as convolutions, matrix
multiplication, FFTs, reductions, stencil updates, particle/event processing,
and iterative solvers. Derive symbolic FLOP formulas from operator counts and
shapes.

Distinguish:

- algorithmic useful FLOPs
- implementation-executed FLOPs when padding or fixed-size batches are evident
- unknown runtime-dependent work, such as convergence-controlled iterations

Do not infer achieved FLOP/s or GPU utilization from theoretical FLOPs.

### 6. Characterize major I/O

Identify major reads, writes, serialization, checkpoint, and host-device transfer
boundaries that are visible in the application. Derive byte formulas using
element counts and data types where possible.

Distinguish storage I/O from host-device transfer and in-device memory traffic.
Do not claim an exact memory-traffic formula unless the code provides enough
evidence.

### 7. Describe synthetic-input requirements

The agent does not create synthetic input in this stage. It records what a later
planner/generator must preserve, including:

- shape and dtype
- value ranges or distributions
- sparsity and masks
- metadata and file format
- cross-field constraints
- semantic invariants that affect control flow or convergence

### 8. Validate the draft

Before presenting the draft, check:

- all formula symbols are defined
- units are dimensionally consistent
- each included input affects at least one modeled quantity or has a clear reason
- every high-confidence claim has source evidence
- uncertain loop counts and branches are explicit
- compute and I/O totals do not double-count phases
- hardware-specific peak specifications are not used as workload FLOPs

## Evidence and Confidence Rules

Evidence types, from strongest to weakest, are:

1. source operator plus statically traceable shape or loop bound
2. application-authored formula or documentation consistent with source
3. publication describing the implemented algorithm
4. framework convention with implementation evidence
5. analogy or agent inference

Confidence values:

- `high`: directly traceable in source
- `medium`: formula is supported but includes explicit assumptions
- `low`: incomplete evidence; human confirmation is required
- `unknown`: cannot be resolved statically

The agent must never present a low-confidence FLOP estimate as an exact value.

## Outputs

Each analysis produces:

```text
application_characterization/<analysis_id>/
  application_characterization.yaml
  analysis_report.md
  human_review.yaml
```

The YAML artifact is the machine-readable contract. The report explains the
application, candidate inputs, dependency formulas, evidence, and questions.
The review file captures human decisions without overwriting the original draft.

The formal output outline is in `application_characterization_schema.yaml`.

## Out of Scope for the First Agent

- executing expensive benchmarks
- profiling FLOPs or hardware counters
- measuring latency, power, or energy
- predicting performance from GPU peak specifications
- fitting empirical resource coefficients
- generating synthetic input before human input confirmation
- integrating coefficients into SystemFlow

## Completion Criterion

The first agent is complete for an application when a researcher has approved:

- the application entry point and analysis boundary
- the important inputs
- major compute and I/O phases
- parameterized compute and I/O formulas
- assumptions and unresolved runtime-dependent quantities
- synthetic-input requirements for the next planning stage

