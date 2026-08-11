# Application Characterization: pty-chi

Status: `approved`

## Summary

Ptychographic reconstruction package. This revised characterization covers the PIE path with square diffraction patterns, one coherent probe mode, input reads, repeated diffraction host-to-device transfers, and iterative reconstruction. Output copying and saving are excluded.

## Candidate Inputs

| Input | Class | Model input | Confidence | Affects |
|---|---|---:|---|---|
| Number of diffraction images | problem_shape | yes | high | PIE iteration FLOPs, PIE minibatch update count, diffraction input bytes, position input bytes, repeated diffraction host-to-device bytes |
| Square image resolution | problem_shape | yes | high | FFT FLOPs, elementwise PIE FLOPs, diffraction input bytes, probe input bytes, repeated diffraction host-to-device bytes |
| Requested batch size | execution_parameter | yes | high | requested CLI configuration, no selected PIE workload quantity because the entry point overrides it with an effective batch size of one |
| Reconstruction iterations | algorithm_parameter | yes | high | total PIE reconstruction FLOPs, PIE update count, repeated diffraction host-to-device bytes |
| Diffraction element width | operational_parameter | no | high | diffraction storage-read bytes, repeated diffraction host-to-device bytes |
| Stored probe element width | operational_parameter | no | high | probe storage-read bytes |
| Stored position-coordinate width | operational_parameter | no | high | position storage-read bytes |

## Compute Model

- `pie_reconstruction_flops`: `R*I*F_PIE`
- `pie_update_count`: `I*Q`
- `preprocessing_runtime_work`: `None`
- `pie_implementation_executed_flops`: `None`

## I/O Model

- `diffraction_hdf5_read`: `R*N^2*b_I`
- `single_mode_probe_hdf5_read`: `N^2*b_Cin`
- `positions_hdf5_read`: `2*R*b_P`
- `pixel_size_scalar_read`: `None`
- `repeated_diffraction_h2d`: `I*R*N^2*b_I`

## Human Decisions Requested

