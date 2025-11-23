# PINN Model Assets

This directory stores deployable artifacts for the PINN-based force estimator:

- `force_estimation_script3.pt`: the TorchScript model exported from training (`scripts/pinn_data_recorder.py` + deepxde pipeline). This file is not versioned by default; copy the latest `.pt` artifact here so that the package can install it into `${prefix}/share/payload_mpc_controller/models/`.
- `normalizer_stats.json`: Min/Max/Mean/Std statistics used to reproduce the exact feature scaling at inference time. A placeholder file is provided for development convenience, but you **must** regenerate it from real datasets via `rosrun payload_mpc_controller compute_normalizer_stats.py`.

Example command to rebuild the statistics after recording datasets into `plots/pinn_dataset/`:

```bash
rosrun payload_mpc_controller compute_normalizer_stats.py \\
  plots/pinn_dataset/pinn_dataset_*.csv \\
  --output $(rospack find payload_mpc_controller)/models/normalizer_stats.json \\
  --pickle  $(rospack find payload_mpc_controller)/models/normalizer_stats.pkl \\
  --pretty
```

During `catkin_make install` or `catkin build --install`, any existing model/stat files in this folder are installed under `share/payload_mpc_controller/models/`, keeping runtime deployments self-contained.
