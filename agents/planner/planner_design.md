# Experiment Planning Agent

## Purpose

The Planning Agent converts one approved application characterization into a
reviewable experiment contract for system-level performance and energy
modeling. It does not compare applications, run benchmarks, fit models, or
silently change approved workload formulas.

## Input

- approved `application_characterization.yaml`
- optional user planning context
- application source code through read-only tools when details must be checked
- planning defaults from `config.toml`

The characterization supplies the approved model inputs, symbolic FLOP and I/O
formulas, execution boundary, assumptions, and synthetic-input requirements.
Algorithms are carried as separate experiment/model groups. Every group uses
the same approved input-space design, and downstream modeling fits independent
coefficients for each algorithm.

## Responsibilities

1. Select sweep, fixed, and invariance-check variables.
2. Define an intended prediction domain and pilot sample points.
3. Avoid blindly expanding every candidate input into a Cartesian product.
4. Select a subset of the supported accelerator catalog.
5. Define timing, power, energy, throughput, and memory measurements.
6. Carry forward the approved system boundary.
7. Produce a synthetic-input specification, not large datasets.
8. Keep the total run count within the configured budget.
9. Request human decisions for assumptions and unresolved choices.

## Deterministic Boundary

The model proposes `matrix_design.base_points`. Local Python code then expands:

```text
algorithm groups x base points x hardware targets x repetitions
```

It assigns run IDs, writes `dry_run_matrix.csv`, and computes the total run
count. The model does not execute applications or submit jobs.

## Outputs

```text
agents/planner/output/
  experiment_plan.yaml
  dry_run_matrix.csv
  planning_report.md
  human_review.yaml
```

## Human Gate

The first plan must remain `awaiting_human_review`. A researcher confirms the
prediction domain, sweep values, fixed values, invariance checks, hardware,
repetitions, measurement boundary, metrics, synthetic-input requirements, and
estimated cost before the Benchmark Runner may consume the plan.
