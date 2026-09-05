"""Generate evaluator configs for the inference-time noise decomposition.

One trained stochastic checkpoint (RF01 = aug26 E01) is rolled out under
several noise regimes that share the weights and differ only in how the
conditioning noise is generated:

    off     scale 0      the learned deterministic backbone g(x, 0)
    fresh   scale 1      training behaviour: new noise every step
    fixed   scale 1      one noise field per trajectory, held through time
    half    scale 0.5    post-hoc amplitude calibration, downward
    double  scale 2.0    post-hoc amplitude calibration, upward
    ens4    fresh, 4 members per IC   ensemble-mean vs member statistics
    mean8   mode mean, 8 draws/step  the conditional-mean operator E_Z g(x,Z) iterated

Every run uses the same held-out 2040s initial conditions and the same
evaluator seed, so "fresh" and "fixed" share their first draw.

This is the study's own generator and stays as the record of what was run.
For the campaign's evaluation use `../../make_eval_config.py`, which reads the
dataset block out of the training template, does the IC-divisibility
arithmetic, narrows the file glob to the reachable years, and exposes this
same noise ladder as `--noise off|fresh|half|fixed|mean`.

Usage:
    python make_eval.py <out_root> <ckpt> <name> --scale S --mode fresh|fixed \
        [--members N] [--steps 1460] [--ics all|4] [--seed 1]
"""

import argparse
import os

import yaml

DATA = dict(
    data_path="/global/cfs/cdirs/e3smdata/simulations/v3.LR.historical_0101.aigo/run",
    # 2040-2049 only: the held-out ICs and their rollouts live here, and
    # opening all 1501 monthly files costs minutes per run.
    file_pattern="v3.LR.historical_0101.aigo.eam.h0.204*.nc",
    rename={
        "PRECT": "surface_precipitation_rate",
        "PRECST": "frozen_precipitation_rate",
        "co2vmr": "global_mean_co2",
    },
    reference_pressure_name="P0",
    overwrite={
        "multiply_scalar": {
            "surface_precipitation_rate": 1000.0,
            "frozen_precipitation_rate": 1000.0,
        }
    },
)

# The template's held-out block: 16 ICs, 2040-2047, January and July.
TEST_ICS = [f"{y}-{m}-03T12:00:00" for y in range(2040, 2048) for m in ("01", "07")]
FOUR_ICS = [f"{y}-01-03T12:00:00" for y in (2040, 2042, 2044, 2046)]
# Every year once, alternating January and July: 2 ICs per rank on 4 ranks.
EIGHT_ICS = [
    f"{y}-{'01' if y % 2 == 0 else '07'}-03T12:00:00" for y in range(2040, 2048)
]

PLOT_VARS = ["Tat2m", "surface_precipitation_rate", "PS", "FLUT", "U_6", "T_7"]
HIST_VARS = [
    "PS", "TS", "LHFLX", "SHFLX", "surface_precipitation_rate", "FLUT",
    "FSNS", "TAUX", "Qat2m", "Uat10m", "Tat2m", "T_1", "T_4", "T_7",
    "STW_4", "STW_7", "U_1", "U_4", "U_6", "V_6",
]  # fmt: skip
SAVE_VARS = ["Tat2m", "surface_precipitation_rate", "PS"]


def make(out_dir, ckpt, scale, mode, members, steps, ics, seed, save_files, draws=1):
    cfg = {
        "experiment_dir": out_dir,
        "n_forward_steps": steps,
        "forward_steps_in_memory": 20,
        "checkpoint_path": ckpt,
        "logging": {
            "log_to_screen": True,
            "log_to_wandb": False,
            "log_to_file": True,
            "project": "ACE2S-sep26-atm",
            "entity": "e3sm-aig",
        },
        "loader": {
            "start_indices": {"times": ics},
            "dataset": DATA,
            # 0: in-process reads. Forked loader workers stalled in a DVS wait
            # (state D, dvsipc_wait_for_resp) at the same window on three nodes.
            "num_data_workers": 0,
        },
        "n_ensemble_per_ic": members,
        "seed": seed,
        "stepper_override": {"noise": {"scale": scale, "mode": mode, "draws": draws}},
        "data_writer": {
            "save_prediction_files": save_files,
            "save_monthly_files": False,
            "names": SAVE_VARS,
        },
        "aggregator": {
            "histogram": {"enabled": True, "variables": HIST_VARS},
            "zonal_mean": {"enabled": False},
            "video": {"enabled": False},
            "trend": {"enabled": False},
            "seasonal": {"enabled": False},
            "near_zero_fraction": {"enabled": False},
            "enso_coefficient": {"enabled": False},
            "ipo_index": {"enabled": False},
            "step_diagnostics": {"correction_scalars": True, "correction_maps": False},
            "time_mean_denorm": {"plot_variables": PLOT_VARS},
            "time_mean_norm": {"target": "norm", "plot_variables": PLOT_VARS},
            "power_spectrum": {"plot_variables": PLOT_VARS},
            "ensembles": [
                {"step": s, "strict": False, "target": t}
                for s in (1, 4, 20, 120, 360, 1460)
                for t in ("denorm", "norm")
            ]
            if members > 1
            else [],
            "step_means": [{"step": s} for s in (1, 4, 20, 120, 360, 1460)],
        },
    }
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out_root")
    ap.add_argument("ckpt")
    ap.add_argument("name")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--mode", default="fresh", choices=["fresh", "fixed", "mean"])
    ap.add_argument("--draws", type=int, default=1)
    ap.add_argument("--members", type=int, default=1)
    ap.add_argument("--steps", type=int, default=1460)
    ap.add_argument("--ics", default="all", choices=["all", "8", "4"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-files", action="store_true")
    a = ap.parse_args()
    out_dir = os.path.join(a.out_root, a.name)
    os.makedirs(out_dir, exist_ok=True)
    ics = {"all": TEST_ICS, "8": EIGHT_ICS, "4": FOUR_ICS}[a.ics]
    cfg = make(
        out_dir, a.ckpt, a.scale, a.mode, a.members, a.steps, ics, a.seed,
        not a.no_files, a.draws,
    )  # fmt: skip
    path = os.path.join(out_dir, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(path)
