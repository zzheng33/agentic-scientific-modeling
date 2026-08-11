# Experiment Plan: pty-chi-pie-pilot-2026-07-30-01

Status: `awaiting_human_review`

Reduced 12-point pilot for fitting end-to-end performance and accelerator-energy models for the approved pty-chi PIE path on one A100. The design sweeps R, N, and I over assumed pilot domains and uses a matched three-level check to verify that requested batch size B is invariant because PIE forces an effective batch size of one.

## Variables

| Input | Role | Strategy | Values / fixed |
|---|---|---|---|
| image_count | sweep | Log-spaced boundary and interior pilot levels over an assumed prediction domain. | [64, 256, 1024] |
| resolution | sweep | Selected power-of-two resolution levels spanning boundaries and interiors. | [32, 64, 128, 256] |
| requested_batch_size | invariance_check | Matched three-level check at identical R, N, and I; use the characterized CLI default at all other points. | [1, 32, 1000] |
| iterations | sweep | Logarithmic boundary and interior levels, excluding zero from this pilot so throughput remains meaningful. | [1, 10, 100] |

## Matrix

- Base points: 12
- Hardware targets: 1
- Repetitions: 3
- Total runs: 36

## Human Decisions Requested

- Approve or revise the assumed pilot prediction domains R=64..1024, N=32..256, and I=1..100.
- Confirm that requested batch levels B={1,32,1000} are sufficient for the PIE invariance check.
- Confirm use of deterministic physically simulated coherent diffraction and approve the planned float32 diffraction, complex64 probe, and float32 position dtypes.
- Provide or approve the exact synthetic object, probe, scan-spacing, and intensity-scaling rules needed for numerically stable PIE runs.
- Confirm the repository-specific --dataset naming and file-layout convention.
- Choose whether the intended latency regime is warmed filesystem cache after one warmup or an explicitly controlled cold-cache regime.
- Approve the A100 as the sole pilot hardware target and confirm that a 20 ms power sampling interval is acceptable.

## Validation

- **WARNING PILOT_DOMAIN_ASSUMED**: The approved characterization gives lower validity bounds but no intended upper prediction bounds. This draft assumes R=64..1024, N=32..256, and I=1..100 for the pilot; human review is required before these are treated as the prediction domain.
- **INFO B_INVARIANCE_CHECK_PLANNED**: B is retained as an approved input but is tested only through a matched three-point invariance check because the selected PIE entry point forces B_eff=1.
- **WARNING SYNTHETIC_PHYSICS_REQUIRES_CONFIRMATION**: The draft specifies deterministic physically coherent synthetic diffraction, but exact object, probe, scan, and numerical-stability parameters remain subject to human approval.
- **WARNING DATASET_LOOKUP_UNRESOLVED**: The CLI-specific mapping from --dataset to the two planned HDF5 files must be verified before a runnable matrix can be produced.
- **WARNING CACHE_POLICY_REVIEW_REQUIRED**: One complete warmup is planned, which may leave HDF5 data in the operating-system page cache. The draft therefore measures a warmed-cache end-to-end regime unless the reviewer requests explicit cold-cache measurements.
- **INFO POWER_SCOPE_ACCELERATOR_ONLY**: Energy is obtained by integrating A100 board power over the approved end-to-end boundary. Host CPU, memory, and storage energy are outside the requested accelerator-power measurement.
- **INFO OUTPUT_BOUNDARY_PRESERVED**: All points use --no-save-output. Output tensor copying and storage writes remain excluded from timing, energy, and byte accounting.
- **INFO RUN_COUNT_WITHIN_LIMIT**: The deterministic run count is 12 base points times 1 hardware target times 3 measured repetitions, totaling 36 measured runs. Warmups are not measured repetitions and are not included in this run-count formula.
