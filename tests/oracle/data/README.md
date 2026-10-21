# Oracle fixtures

`test_oracle.py` checks this fork's tracking against the **oracle**: the standalone
Kinefly, a line-for-line Python 3 port of the 2016 ROS code with unchanged algorithms
(repo `isabel-hess/Kinefly`, branch `standalone-video`). The fixtures were made with
its commit `e4143fb`.

| File | What it is |
|---|---|
| `v001a.mp4` | `bias_video_cam_0_date_2026_08_25_time_14_10_14_v001a.mp4`, unmodified: 1440×1080 H.264, 574 frames, a tethered fly walking with folded wings. |
| `handles_1440x1080.yaml` | Handle positions in the oracle's format (`docs/ros2-fork/oracle_handles_1440x1080.yaml` in the standalone repo). |
| `rig_example.yaml`, `rig_walking.yaml` | This fork's rig configs, equivalent to the oracle's `params_example.yaml` and `params_walking.yaml` (`n_edges_max` 1 and 3). |
| `expected_example.csv`, `expected_walking.csv` | The oracle's output on every frame, with its timing columns removed. |

Expected values are compared as text, so "exact" means equal at the CSV's precision
(angles and intensity 6 decimals, radii 4, frequency 3).

## Regenerating

Only regenerate after an **intended** change to the algorithms, and say so in the commit.
From the standalone repo, in PowerShell:

```powershell
$d = "PATH_TO_THIS_FORK\tests\oracle\data"
foreach ($p in "example", "walking") {
    Copy-Item $d\handles_1440x1080.yaml $env:TEMP\h.yaml   # The oracle saves handles on exit.
    pixi run python nodes/kinefly.py $d\v001a.mp4 --offline --no-gui --scale 1.0 --params params_$p.yaml --gui-yaml $env:TEMP\h.yaml --csv $env:TEMP\ours_$p.csv
}
```

Then drop the oracle's timing columns (`t_acquired`, `t_done`, `latency_ms`,
`process_ms`, `dropped_total`) and save the rest as `expected_<p>.csv`.
