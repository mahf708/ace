"""One-step decomposition of a stochastic checkpoint's transition operator.

For validation states x with truth y = x_{t+1}, compare

    g0 = g(x, 0)            the deterministic backbone (noise off)
    gk = g(x, Z_k)          K fresh-noise draws
    m  = mean_k gk          the conditional mean the stochastic operator implies

and report, per output channel, area-weighted and pooled over states:

    backbone error   <|g0 - y|^2>
    cond-mean error  <|m  - y|^2>   (finite-K corrected)
    member error     mean_k <|gk - y|^2>  = cond-mean error + spread
    spread           tr Sigma = mean_k <|gk - m|^2>  (finite-K corrected)
    noise drift      <|m - g0|^2>
    alignment A      <(m - g0) . (y - g0)> / (|m - g0| |y - g0|)
    persistence      <|x - y|^2>
    MAE(g0) vs fair CRPS of the K-member ensemble

A(x) > 0 means the noise-induced (Jensen) drift of the mean moves the
backbone toward the truth. Outputs are post-corrector, exactly what the
model emits in a rollout.

Usage: python one_step_drift.py <ckpt> <out.npz> [--k 16] [--batches 32]
"""

import argparse
import logging

import dacite
import numpy as np
import torch

from fme.ace.data_loading.config import DataLoaderConfig
from fme.ace.data_loading.getters import get_gridded_data
from fme.ace.stepper import NoiseOverrideConfig, load_stepper
from fme.core.coordinates import LatLonCoordinates
from fme.core.distributed import Distributed
from fme.core.ensemble import get_crps

DATA = dict(
    data_path="/global/cfs/cdirs/e3smdata/simulations/v3.LR.historical_0101.aigo/run",
    # 1990-1999 only: opening all 1501 monthly files costs minutes per run.
    file_pattern="v3.LR.historical_0101.aigo.eam.h0.199*.nc",
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
    subset={"start_time": "1990-01-01T06:00:00", "stop_time": "1995-01-01T00:00:00"},
)


def wmean(t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Area-weighted mean over the trailing (lat, lon) dims."""
    return (t * w).sum(dim=(-2, -1)) / w.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--batches", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=4)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(0)

    stepper = load_stepper(a.ckpt)
    stepper.set_eval()
    cfg = stepper.config
    req = cfg.get_evaluation_window_data_requirements(n_forward_steps=1)
    loader_cfg = dacite.from_dict(
        DataLoaderConfig,
        {"dataset": DATA, "batch_size": a.batch_size, "num_data_workers": 4},
        config=dacite.Config(strict=True),
    )
    data = get_gridded_data(loader_cfg, train=True, requirements=req)
    out_names = list(cfg.output_names)
    prog_names = list(cfg.prognostic_names)
    hc = data.dataset_info.horizontal_coordinates
    # an analysis script on a lat-lon dataset: LatLonCoordinates is the only case
    assert isinstance(hc, LatLonCoordinates)
    lat = hc.lat.detach().float()
    device = next(stepper.modules[0].parameters()).device
    w = torch.cos(torch.deg2rad(lat))[:, None].to(device)

    K = a.k
    keys = ("e0", "em", "ek", "tr", "drift", "pers", "num", "d0", "dy", "mae0", "crps")
    acc = {key: torch.zeros(len(out_names), dtype=torch.float64) for key in keys}
    per_sample_A = []
    n_states = 0
    for i, batch in enumerate(data.loader):
        if i >= a.batches:
            break
        batch = batch.to_device()
        ic = batch.get_start(prognostic_names=prog_names, n_ic_timesteps=1)
        y = torch.stack([batch.data[n][:, 1] for n in out_names], dim=1)  # (B,C,H,W)
        x = torch.stack(
            [
                batch.data[n][:, 0] if n in batch.data else torch.zeros_like(y[:, 0])
                for n in out_names
            ],
            dim=1,
        )

        with torch.no_grad():
            stepper.set_noise_override(NoiseOverrideConfig(scale=0.0))
            g0 = stepper.predict(ic, batch, compute_derived_variables=False)[0]
            g0 = torch.stack([g0.data[n][:, 0] for n in out_names], dim=1)
            stepper.set_noise_override(NoiseOverrideConfig(scale=1.0))
            gs = []
            for _ in range(K):
                gk = stepper.predict(ic, batch, compute_derived_variables=False)[0]
                gs.append(torch.stack([gk.data[n][:, 0] for n in out_names], dim=1))
            G = torch.stack(gs, dim=1)  # (B,K,C,H,W)
        m = G.mean(dim=1)
        spread_raw = ((G - m[:, None]) ** 2).mean(dim=1)  # (B,C,H,W), biased by (K-1)/K
        tr = spread_raw * K / (K - 1)
        em_raw = (m - y) ** 2
        em = em_raw - tr / K  # unbiased estimate of |E g - y|^2
        ek = ((G - y[:, None]) ** 2).mean(dim=1)
        e0 = (g0 - y) ** 2
        d = m - g0
        r = y - g0
        pers = (x - y) ** 2
        # per-sample, per-channel alignment
        num = wmean(d * r, w)
        den = torch.sqrt(wmean(d**2, w) * wmean(r**2, w)) + 1e-30
        per_sample_A.append((num / den).cpu().numpy())
        gen = G.permute(0, 1, 2, 3, 4)  # (B,K,C,H,W) already
        crps = get_crps(gen, y[:, None]).mean(dim=0)  # (C,H,W) -> average over batch
        mae0 = (g0 - y).abs()

        for key, val in (
            ("e0", e0), ("em", em), ("ek", ek), ("tr", tr), ("drift", d**2),
            ("pers", pers), ("num", d * r), ("d0", d**2), ("dy", r**2),
            ("mae0", mae0),
        ):  # fmt: skip
            acc[key] += wmean(val, w).sum(dim=0).double().cpu()
        acc["crps"] += wmean(crps, w).double().cpu() * y.shape[0]
        n_states += y.shape[0]
        logging.info("batch %d done, %d states", i, n_states)

    res = {k: (v / n_states).numpy() for k, v in acc.items()}
    res["A_pooled"] = res["num"] / np.sqrt(res["d0"] * res["dy"])
    A_ps = np.concatenate(per_sample_A, axis=0)  # (N, C)
    np.savez(
        a.out,
        out_names=np.array(out_names),
        n_states=n_states,
        K=K,
        A_per_sample=A_ps,
        **res,
    )
    print(
        f"{'channel':>28} {'e0/pers':>8} {'em/pers':>8} {'ek/pers':>8} {'tr/e0':>7} "
        f"{'drift/e0':>9} {'A_pool':>7} {'A_med':>6} {'A>0':>5} {'crps/mae0':>9}"
    )
    for c, n in enumerate(out_names):
        print(
            f"{n:>28} {res['e0'][c]/res['pers'][c]:8.4f} "
            f"{res['em'][c]/res['pers'][c]:8.4f} "
            f"{res['ek'][c]/res['pers'][c]:8.4f} {res['tr'][c]/res['e0'][c]:7.3f} "
            f"{res['drift'][c]/res['e0'][c]:9.4f} {res['A_pooled'][c]:7.3f} "
            f"{np.median(A_ps[:, c]):6.3f} {(A_ps[:, c] > 0).mean():5.2f} "
            f"{res['crps'][c]/res['mae0'][c]:9.4f}"
        )
    tot = lambda k: float(np.mean(res[k] / res["pers"]))  # noqa: E731
    print(
        f"\nchannel-mean of (err/persistence): backbone {tot('e0'):.4f}  "
        f"cond-mean {tot('em'):.4f}  member {tot('ek'):.4f}"
    )
    tr_e0 = float(np.mean(res["tr"] / res["e0"]))
    drift_e0 = float(np.mean(res["drift"] / res["e0"]))
    print(f"channel-mean tr/e0 {tr_e0:.3f}  drift/e0 {drift_e0:.4f}")
    print(f"channel-mean CRPS/MAE(g0) {float(np.mean(res['crps']/res['mae0'])):.4f}")
    a_pool = float(np.mean(res["A_pooled"]))
    print(
        f"channel-mean pooled alignment A {a_pool:.3f}; "
        f"fraction of (state,channel) with A>0: {(A_ps > 0).mean():.3f}"
    )


if __name__ == "__main__":
    with Distributed.context(handle_signals=False):
        main()
