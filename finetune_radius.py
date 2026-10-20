r"""finetune_radius.py - Einstein radii of the real Euclid lenses estimated with UNet-Radius-mini and Attention
UNet-Radius-mini: the networks applied directly, fine-tuned on the measured radii, and trained on them from random
weights

AttentionRSA.

Measured radii
    The lens models of the Euclid Q1 Strong Lensing Discovery Engine, fitted with PyAutoLens (Zenodo record 15025832,
    file modeling_lens_mass.csv). The measured radius is the median of the posterior of the effective Einstein radius
    (einstein_radius_effective_median_pdf, arcseconds). The models are joined to the lenses of the EUCLID-SAGE
    benchmarks by the lens identifier (id_str in the Zenodo file, lens_id in the benchmark tables). Lenses without a
    model are not used, nor lenses with a radius below --min-radius (0.05"): the effective Einstein radius is 0 when
    the lens model has no tangential critical curve, so such a value does not measure the size of the lens.

Networks (--networks NAME=PATH ...)
    Networks trained on simulations by train_copies.py with --model radius-mini (with or without --attention), for
    example the copies of one network with different initial weights.

1. Networks applied without training on real stamps
    Each network is applied to the real stamps with the input transform of its simulated training images. The
    simulations have pixels of 0.101", the real stamps pixels of 0.1"; a structure that spans the same number of pixels
    is therefore 0.1 / 0.101 times smaller on the sky in a real stamp, and the estimated radius is multiplied by this
    factor.

2. Networks fine-tuned on the measured radii, or trained on them from random weights
    The encoder, the readout and the Einstein radius layer (models.RadiusRegressor) are trained on the measured radii,
    with the absolute error as the loss; the decoder is not used, because the real stamps have no arc masks. The
    regressor starts either from the weights of a network trained on simulations (initialisation pretrained-NAME) or
    from random weights (random; with the architecture and input transform of the first network of --networks, one
    network per training seed). In every case:
    - five folds (seed 42), the same for all networks, so that every lens is tested once; within each fold, 80% of
      the training lenses for training and 20% for validation (seed 42 + fold number). The radii are continuous, so
      the folds are not stratified;
    - for each learning rate of the grid (1e-4, 3e-4, 1e-3, 3e-3): 60 epochs with AdamW, weight decay 1e-4, batches of
      16 stamps, a learning rate that decreases to zero along a cosine, and the weights of the epoch with the lowest
      validation error;
    - the learning rate with the lowest validation error (rounded to four decimals; for equal values, the lower rate)
      is chosen, and its network is tested once on the test lenses of the fold;
    - input: asinh(x / s) with the scale s saved with the network, standardised with the mean and standard deviation
      of the training stamps of the fold; training stamps are randomly rotated and reflected (data.py);
    - the random generators are seeded with 1000 s + k (training seed s, fold k), so that all learning rates of a fold
      start from the same weights and see the training stamps in the same order.

Metrics (over the lenses with a measured radius; error e = estimated - measured radius)
    Mean absolute error with a 95% bootstrap interval (2,000 resamples of the lenses), median absolute error, root mean
    square error, normalised median absolute deviation NMAD = 1.4826 median |e - median(e)|, mean error, median
    relative error |e| / measured radius, Spearman rank correlation of estimated and measured radii, and the mean
    absolute error of grade A and grade B lenses.

Outputs (results/<--out-name>/; <--out-name>_quick/ for a quick run)
    applied_<name>.csv                          per lens: identifiers, grade, measured and estimated radius
    finetuned_<initialisation>_seed<s>.csv      per lens: the estimate of the network that had the lens in its test
                                                fold, the fold and the chosen learning rate
    folds/<initialisation>_seed<s>_fold<k>.json     learning-rate search in fold k (finished folds are
                                                    skipped when the script is run again)
    folds/<initialisation>_seed<s>_fold<k>_network.pt   the tested network of fold k, with the input transform of
                                                    the fold
    measured_radii.json, summary.csv, results.md

Usage
    python finetune_radius.py --measured modeling_lens_mass.csv --seeds 1 --no-random \
        --networks radius-mini-init01=results/simulations/radius-mini-init01/best_unet.pt \
                   radius-mini-init02=results/simulations/radius-mini-init02/best_unet.pt
    python finetune_radius.py --measured modeling_lens_mass.csv --seeds 1 2 3 --no-pretrained \
        --networks radius-mini-init01=results/simulations/radius-mini-init01/best_unet.pt     # random weights
    python finetune_radius.py ... --quick                                     # 2 learning rates, 3 folds, 3 epochs
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader

from data import (RESULTS_FOLDER, StampDataset, asinh_transform, load_stamps_in_simulation_units,
                  read_benchmark_table, set_random_seeds)
from models import UNet, build_regressor, choose_device, load_saved_network

SIMULATION_PIXEL_SCALE = 0.101          # arcseconds per pixel of the simulated images
STAMP_PIXEL_SCALE = 0.1                 # arcseconds per pixel of the real stamps
BENCHMARK = "EUCLID-SAGE-1k-R_VIS"      # all benchmarks contain the same 487 lenses
RESAMPLES = 2000                        # bootstrap resamples for the 95% intervals
BOOTSTRAP_SEED = 20261008
NMAD_FACTOR = 1.4826                    # makes the NMAD equal to the standard deviation for a normal distribution


# ===================================================================================================== 1. data

def lenses_with_measured_radius(measured_path, id_column, radius_column, min_radius):
    """The lenses of the benchmarks that have a valid measured Einstein radius, and statistics of the measured radii."""
    lenses = read_benchmark_table(BENCHMARK)
    lenses = lenses[lenses.label == 1][["stamp_file", "object_id", "lens_id", "grade"]]
    measured = pd.read_csv(measured_path)
    if id_column not in measured or radius_column not in measured:
        raise SystemExit(f"{measured_path} needs the columns {id_column} and {radius_column}; it has "
                         f"{', '.join(measured.columns)}")
    measured = measured[[id_column, radius_column]].dropna(subset=[radius_column]).drop_duplicates(id_column)
    measured = measured.rename(columns={id_column: "lens_id", radius_column: "measured_radius"})
    # Only lenses with a model remain after the join; then the radii below min_radius are left out.
    joined = lenses.merge(measured, on="lens_id", how="inner").reset_index(drop=True)
    too_small = joined.measured_radius < min_radius
    valid = joined[~too_small].reset_index(drop=True)
    statistics = {"radii_in_file": int(len(measured)), "lenses_matched": int(len(joined)),
                  "lenses_below_min_radius": int(too_small.sum()), "min_radius": min_radius,
                  "lenses_used": int(len(valid)), "grade_A": int((valid.grade == "A").sum()),
                  "grade_B": int((valid.grade == "B").sum()),
                  "quantiles": {f"{quantile:g}": round(float(valid.measured_radius.quantile(quantile)), 3)
                                for quantile in (0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)}}
    print(f"{len(joined)} of {len(lenses)} lenses have a measured Einstein radius ({len(measured)} radii in "
          f"{measured_path.name}); {int(too_small.sum())} below {min_radius}\" left out; {len(valid)} used "
          f"({statistics['grade_A']} grade A, {statistics['grade_B']} grade B)", flush=True)
    if len(valid) < 20:                 # usually a sign that the identifiers of the two tables do not match
        raise SystemExit("Too few lenses with a measured radius; check the identifier column.")
    return valid, statistics


# ================================================================================= 2. networks applied unchanged

@torch.no_grad()
def apply_network(saved_network, transformed_stamps, device):
    """Estimated radii of a saved network, converted to the pixel scale of the real stamps."""
    network = UNet(**saved_network["architecture"])
    network.load_state_dict(saved_network["model_state_dict"])
    network.to(device).eval()
    # Standardisation with the mean and standard deviation of the simulated training images of the network.
    preprocessing = saved_network["preprocessing"]
    radii = []
    for start in range(0, len(transformed_stamps), 32):
        batch = (transformed_stamps[start:start + 32] - preprocessing["mean"]) / preprocessing["std"]
        _, radius = network(torch.from_numpy(batch[:, None].astype(np.float32)).to(device),
                            return_einstein_radius=True)
        radii.append(radius.cpu().numpy())
    return np.concatenate(radii) * STAMP_PIXEL_SCALE / SIMULATION_PIXEL_SCALE


# ======================================================================================== 3. fine-tuning

@torch.no_grad()
def mean_absolute_error(network, loader, device):
    """Mean absolute error of the estimated radii over a loader, and the estimates."""
    network.eval()                      # no dropout during evaluation
    errors, estimates = [], []
    for images, radii, _ in loader:
        estimate = network(images.to(device)).cpu()
        errors.append((estimate - radii).abs())
        estimates.append(estimate)
    return float(torch.cat(errors).mean()), torch.cat(estimates).numpy()


def train_one_rate(network, training_loader, validation_loader, learning_rate, settings, device):
    """Train for settings.epochs epochs and keep the weights of the epoch with the lowest validation error."""
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=settings.weight_decay)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(settings.epochs, 1))
    loss_function = nn.L1Loss()                     # mean absolute error of the radius over a batch
    best = {"validation_error": float("inf"), "best_epoch": 0}
    best_weights = copy.deepcopy(network.state_dict())
    for epoch in range(1, settings.epochs + 1):
        network.train()
        for images, radii, _ in training_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(network(images.to(device)), radii.to(device))
            loss.backward()
            optimizer.step()
        schedule.step()
        validation_error, _ = mean_absolute_error(network, validation_loader, device)
        if validation_error < best["validation_error"]:
            best = {"validation_error": validation_error, "best_epoch": epoch}
            best_weights = copy.deepcopy(network.state_dict())
    network.load_state_dict(best_weights)
    return best


def run_fold(fold, training_rows, test_rows, transformed, radii, saved_network, initialisation, seed, settings, device,
             network_path):
    """Chooses the learning rate on the validation lenses of one fold and tests the chosen network once on the test
    lenses. The tested network is saved to network_path."""
    pretrained = initialisation != "random"
    # 80% of the training lenses of the fold for training, 20% for validation; the standardisation uses the training
    # lenses only.
    fit_rows, validation_rows = train_test_split(training_rows, test_size=0.2, random_state=settings.split_seed + fold)
    mean, standard_deviation = transformed[fit_rows].mean(), transformed[fit_rows].std()
    radii = radii.astype(np.float32)

    def loader(rows, augment, shuffle):
        """Batches of the given lenses; augment: random rotations and reflections; shuffle: random order."""
        return DataLoader(StampDataset(transformed[rows], radii[rows], mean, standard_deviation, augment),
                          batch_size=settings.batch_size, shuffle=shuffle)

    # One network per learning rate, each from the same random weights and order of the training lenses.
    candidates = []
    for learning_rate in settings.learning_rates:
        set_random_seeds(1000 * seed + fold)
        network = build_regressor(saved_network, pretrained).to(device)
        best = train_one_rate(network, loader(fit_rows, True, True), loader(validation_rows, False, False),
                              learning_rate, settings, device)
        candidates.append({"learning_rate": learning_rate, **best, "network": network})
        print(f"      fold {fold}, learning rate {learning_rate:g}: best epoch {best['best_epoch']}, validation "
              f"error {best['validation_error']:.4f}\"", flush=True)
    chosen = min(candidates, key=lambda candidate: (round(candidate["validation_error"], 4),
                                                    candidate["learning_rate"]))
    test_error, estimates = mean_absolute_error(chosen["network"], loader(test_rows, False, False), device)
    torch.save({"model_state_dict": chosen["network"].cpu().state_dict(),
                "architecture": saved_network["architecture"], "initialisation": initialisation, "seed": seed,
                "fold": fold, "learning_rate": chosen["learning_rate"], "best_epoch": chosen["best_epoch"],
                "preprocessing": {"asinh_scale": float(saved_network["preprocessing"]["asinh_scale"]),
                                  "mean": float(mean), "std": float(standard_deviation)}},
               network_path)
    return {"fold": fold, "seed": seed, "chosen_learning_rate": chosen["learning_rate"],
            "chosen_best_epoch": chosen["best_epoch"], "chosen_validation_error": chosen["validation_error"],
            "candidates": [{key: value for key, value in candidate.items() if key != "network"}
                           for candidate in candidates],
            "test_rows": [int(row) for row in test_rows], "test_estimates": [float(estimate) for estimate in estimates],
            "test_error": test_error}


def fine_tune(lenses, transformed, saved_network, initialisation, seed, settings, device, output_folder):
    """Estimates for all lenses, each from the network that had the lens in its test fold, for one initialisation
    ("random" or "pretrained-<name>") and one training seed."""
    radii = lenses.measured_radius.to_numpy(dtype=float)
    fold_folder = output_folder / "folds"
    fold_folder.mkdir(parents=True, exist_ok=True)
    estimates = np.full(len(lenses), np.nan)
    chosen = np.full(len(lenses), np.nan)
    folds = np.zeros(len(lenses), dtype=int)
    splitter = KFold(n_splits=settings.folds, shuffle=True, random_state=settings.split_seed)
    for fold, (training_rows, test_rows) in enumerate(splitter.split(radii), start=1):
        path = fold_folder / f"{initialisation}_seed{seed}_fold{fold}.json"
        network_path = fold_folder / f"{initialisation}_seed{seed}_fold{fold}_network.pt"
        if path.exists():                            # a fold finished in an earlier, interrupted run
            record = json.loads(path.read_text())
            print(f"    fold {fold}: finished earlier", flush=True)
        else:
            start = time.time()
            record = run_fold(fold, training_rows, test_rows, transformed, radii, saved_network, initialisation, seed,
                              settings, device, network_path)
            record["init"] = initialisation
            record["runtime_minutes"] = round((time.time() - start) / 60, 1)
            path.write_text(json.dumps(record, indent=1))
            print(f"    fold {fold}: learning rate {record['chosen_learning_rate']:g}, best epoch "
                  f"{record['chosen_best_epoch']}, test error {record['test_error']:.3f}\", "
                  f"{record['runtime_minutes']:.1f} min", flush=True)
        rows = np.array(record["test_rows"])
        estimates[rows], chosen[rows], folds[rows] = record["test_estimates"], record["chosen_learning_rate"], fold
    table = lenses[["lens_id", "object_id", "grade", "measured_radius"]].copy()
    table["estimated_radius"], table["fold"], table["learning_rate"] = estimates.round(4), folds, chosen
    table.to_csv(output_folder / f"finetuned_{initialisation}_seed{seed}.csv", index=False)
    return estimates


# ============================================================================================== 4. metrics

def error_summary(name, estimates, measured, grades, generator):
    """The metrics of one set of estimated radii (see the description at the top)."""
    estimates = np.asarray(estimates, dtype=float)
    error = estimates - measured                     # positive: the radius is overestimated
    absolute = np.abs(error)
    valid = np.isfinite(absolute)
    resampled = [absolute[valid][generator.integers(0, valid.sum(), valid.sum())].mean() for _ in range(RESAMPLES)]
    row = {"estimate": name, "lenses": int(valid.sum()), "mean_absolute_error": float(absolute[valid].mean()),
           "mae_ci_low": float(np.percentile(resampled, 2.5)), "mae_ci_high": float(np.percentile(resampled, 97.5)),
           "median_absolute_error": float(np.median(absolute[valid])),
           "rmse": float(np.sqrt(np.mean(error[valid] ** 2))),
           "nmad": float(NMAD_FACTOR * np.median(np.abs(error[valid] - np.median(error[valid])))),
           "mean_error": float(error[valid].mean()),
           "median_relative_error": float(np.median(absolute[valid] / measured[valid])),
           "spearman": float(spearmanr(estimates[valid], measured[valid])[0])}
    for grade in ("A", "B"):
        selected = valid & (grades == grade)
        row[f"mae_grade_{grade}"] = float(absolute[selected].mean()) if selected.any() else float("nan")
    return {key: (round(value, 4) if isinstance(value, float) else value) for key, value in row.items()}


def write_report(statistics, radius_column, summary, output_folder):
    """results.md: the measured radii and the metrics of all estimates."""
    quantiles = statistics["quantiles"]
    lines = ["# Einstein radii of the real lenses", "",
             f"Measured radii: PyAutoLens lens models ({radius_column}). {statistics['lenses_used']} lenses "
             f"({statistics['grade_A']} grade A, {statistics['grade_B']} grade B); quantiles (0, 5, 25, 50, 75, 95, "
             f"100%): {', '.join(str(quantiles[key]) for key in quantiles)}\". Brackets: 95% bootstrap interval.", "",
             "| Estimate | Lenses | Mean absolute error | Median absolute error | RMSE | NMAD | Mean error | "
             "Median relative error | Spearman | Grade A | Grade B |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {row['estimate']} | {row['lenses']} | {row['mean_absolute_error']:.3f}\" "
                     f"[{row['mae_ci_low']:.3f}, {row['mae_ci_high']:.3f}] | {row['median_absolute_error']:.3f}\" | "
                     f"{row['rmse']:.3f}\" | {row['nmad']:.3f}\" | {row['mean_error']:+.3f}\" | "
                     f"{100 * row['median_relative_error']:.0f}% | {row['spearman']:.2f} | "
                     f"{row['mae_grade_A']:.3f}\" | {row['mae_grade_B']:.3f}\" |")
    (output_folder / "results.md").write_text("\n".join(lines) + "\n")


# ========================================================================================= 5. command line

def parse_networks(entries):
    """NAME=PATH entries; without NAME=, the name of the folder of the network file is used."""
    networks = {}
    for entry in entries:
        name, _, path = entry.partition("=") if "=" in entry else (Path(entry).parent.name, "", entry)
        networks[name] = Path(path)
    return networks


def main():
    argument_parser = argparse.ArgumentParser(description="Einstein radii of the real lenses")
    argument_parser.add_argument("--measured", type=Path, required=True,
                                 help="modeling_lens_mass.csv, the lens models of the Q1 Strong Lensing "
                                      "Discovery Engine (Zenodo record 15025832)")
    argument_parser.add_argument("--networks", nargs="+", required=True,
                                 help="NAME=PATH of networks trained on simulations with --model radius-mini")
    argument_parser.add_argument("--id-column", default="id_str", help="column of the lens identifier in --measured")
    argument_parser.add_argument("--radius-column", default="einstein_radius_effective_median_pdf",
                                 help="column of the measured Einstein radius (arcseconds) in --measured")
    argument_parser.add_argument("--min-radius", type=float, default=0.05,
                                 help="smallest measured radius used (default 0.05\")")
    argument_parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-4, 3e-4, 1e-3, 3e-3],
                                 help="grid of initial learning rates searched in every fold")
    argument_parser.add_argument("--epochs", type=int, default=60, help="training epochs per learning rate")
    argument_parser.add_argument("--batch-size", type=int, default=16, help="stamps per batch")
    argument_parser.add_argument("--weight-decay", type=float, default=1e-4, help="weight decay of AdamW")
    argument_parser.add_argument("--folds", type=int, default=5, help="number of cross-validation folds")
    argument_parser.add_argument("--split-seed", type=int, default=42,
                                 help="seed of the folds and of the training-validation splits")
    argument_parser.add_argument("--seeds", nargs="+", type=int, default=[1], help="training seeds")
    argument_parser.add_argument("--no-random", action="store_true", help="leave out the random initialisation")
    argument_parser.add_argument("--no-pretrained", action="store_true",
                                 help="only the random initialisation, one network per training seed; --networks then "
                                      "gives only the architecture and input transform")
    argument_parser.add_argument("--out-name", default="einstein_radius_real", help="name of the results folder")
    argument_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto",
                                 help="auto: a GPU if one is available, otherwise the CPU")
    argument_parser.add_argument("--quick", action="store_true", help="2 learning rates, 3 folds and 3 epochs")
    settings = argument_parser.parse_args()
    if settings.quick:
        settings.learning_rates, settings.folds, settings.epochs = [3e-4, 1e-3], 3, 3
    networks = parse_networks(settings.networks)
    saved_networks = {}
    for name, path in networks.items():
        saved_networks[name] = load_saved_network(path)
        if not saved_networks[name]["architecture"].get("einstein_radius", False):
            raise SystemExit(f"{path} has no Einstein radius output (--model radius-mini expected)")
    first_network = next(iter(networks))    # its architecture and input transform are used for random weights
    device = choose_device(settings.device)
    output_folder = RESULTS_FOLDER / (settings.out_name + ("_quick" if settings.quick else ""))
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Networks: {', '.join(f'{name} ({path})' for name, path in networks.items())}; device {device}; results "
          f"in {output_folder}", flush=True)

    lenses, statistics = lenses_with_measured_radius(settings.measured, settings.id_column, settings.radius_column,
                                                     settings.min_radius)
    (output_folder / "measured_radii.json").write_text(json.dumps(statistics, indent=1))
    stamps = load_stamps_in_simulation_units(lenses)
    transformed = {name: asinh_transform(stamps, float(saved["preprocessing"]["asinh_scale"]))
                   for name, saved in saved_networks.items()}
    measured = lenses.measured_radius.to_numpy(dtype=float)
    grades = lenses.grade.to_numpy()
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    summary = []

    initialisations = (([] if settings.no_pretrained else [f"pretrained-{name}" for name in networks])
                       + ([] if settings.no_random else ["random"]))
    if not initialisations:
        raise SystemExit("--no-pretrained and --no-random together leave nothing to fine-tune")
    if not settings.no_pretrained:
        print("\n1. Networks applied without training on real stamps", flush=True)
        for name, saved in saved_networks.items():
            applied = apply_network(saved, transformed[name], device)
            table = lenses[["lens_id", "object_id", "grade", "measured_radius"]].copy()
            table["estimated_radius"] = applied.round(4)
            table.to_csv(output_folder / f"applied_{name}.csv", index=False)
            summary.append(error_summary(f"applied {name}", applied, measured, grades, generator))
            print(f"  {name}: mean absolute error {summary[-1]['mean_absolute_error']:.3f}\", mean error "
                  f"{summary[-1]['mean_error']:+.3f}\"", flush=True)

    print("\n2. Networks fine-tuned on the measured radii", flush=True)
    for seed in settings.seeds:
        for initialisation in initialisations:
            name = initialisation[len("pretrained-"):] if initialisation != "random" else first_network
            print(f"  {initialisation}, seed {seed}", flush=True)
            estimates = fine_tune(lenses, transformed[name], saved_networks[name], initialisation, seed, settings,
                                  device, output_folder)
            summary.append(error_summary(f"fine-tuned {initialisation}, seed {seed}", estimates, measured, grades,
                                         generator))
            print(f"  {initialisation}, seed {seed}: mean absolute error {summary[-1]['mean_absolute_error']:.3f}\", "
                  f"median {summary[-1]['median_absolute_error']:.3f}\"", flush=True)

    pd.DataFrame(summary).to_csv(output_folder / "summary.csv", index=False)
    write_report(statistics, settings.radius_column, summary, output_folder)
    print(f"\n{(output_folder / 'results.md').read_text()}\nResults saved in {output_folder}", flush=True)


if __name__ == "__main__":
    main()
