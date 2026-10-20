r"""compute_rdms.py - the representational dissimilarity matrix (RDM) of every layer of every saved network of one
group, on the simulated test images and on the real images held out from fine-tuning

AttentionRSA.

Runs on a CPU (or a GPU, --device cuda or mps) after the networks have been trained; trains nothing.

Inputs
    --results     the results folder of one group of networks (for example results/rsa_radius/), with the folders
                  written by train_copies.py, finetune_radius.py and finetune_lens.py (see reproduce.sh); folders of a
                  quick run (ending in _quick) are found as well
    --simulated   the simulated set on which the networks were trained (simulate.py)
    --benchmarks  the folder with the 1k tables and stamps/ (benchmarks/ of the EUCLID-SAGE repository; default
                  EUCLID_SAGE_BENCHMARK_DIR)
    --tables      the folder with the 1k tables, if it is not --benchmarks

Networks (kinds; UNet-Radius-mini and UNet-Starlight include their attention versions)
    simulation  simulations/<network>-init<k>/best_unet.pt: the copies trained on simulations
    finetuned   UNet-Radius-mini: einstein_radius_real/folds/pretrained-<copy>_seed1_fold<f>_network.pt
                UNet-Starlight:   fine_tuning_1k_D/<copy>/EUCLID-SAGE-1k-D_VIS/pretrained/seed1/fold<f>_network.pt
    random      UNet-Radius-mini: einstein_radius_real_random_weights/folds/random_seed<s>_fold<f>_network.pt
                UNet-Starlight:   random_weights_1k_D/<network>/EUCLID-SAGE-1k-D_VIS/random/seed<s>/fold<f>_network.pt
    (the folders written by finetune_radius.py and finetune_lens.py with the names used in reproduce.sh)

Images
    simulated   the test images of the split (split_indices.npz, the same for every copy), seen by no network
    real        UNet-Radius-mini: the lenses with a measured Einstein radius; UNet-Starlight: the 1,000 stamps of
                EUCLID-SAGE-1k-D.
                A network of fold f (fine-tuned, or trained from random weights) is applied only to the stamps of
                fold f, which it did not see in training; the networks trained only on simulations are applied to all
                of them.
    Every network is applied with its own input transform (asinh scale, mean and standard deviation).

Layers
    inc, d1, d2, d3, d4 (the encoder; d4 is the 6 x 6 bottleneck), readout (the average of the bottleneck, weighted
    by the attention map in the attention networks), and the output (radius, or lens_logits). For the networks
    trained on simulations also the decoder, u1 ... u4, and the arc logits (out).

RDM (representational dissimilarity matrix; Kriegeskorte et al. 2008)
    For n images, the n x n matrix of the dissimilarities between the activations of a layer for every pair of
    images: 1 - Pearson correlation over the flattened feature maps (--features pooled: over the average of every
    feature map over its pixels). The correlation does not depend on the overall level and spread of the activations
    of an image, only on their pattern. For layers with one or two values per image (radius, lens logits) a
    correlation is undefined or trivial, and the Euclidean distance is used. The RDMs are symmetric with zeros on the
    diagonal and are saved as their upper triangles (condensed).

Outputs (--out, default rdms/<group folder>/)
    <image set>/<network id>.npz  rdm_<layer> (condensed, float32), rows (the image rows: test indices of the
                                  simulated set, or rows of the real table), outputs, and for the attention networks
                                  attention (the 6 x 6 weights of every image)
    networks.csv                  one row per network and image set
    real_images.csv               the real images: row, stamp, label, grade, fold and, for UNet-Radius-mini,
                                  the measured radius

Usage
    python compute_rdms.py --results results/rsa_radius --simulated radius_training_10000.npz \
        --benchmarks ../EUCLID-SAGE/benchmarks
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import squareform

from data import BENCHMARK_FOLDER, asinh_transform, load_stamps_in_simulation_units
from models import ENCODER_BLOCKS, EncoderClassifier, RadiusRegressor, UNet, pool_bottleneck

DECODER_BLOCKS = ("u1", "u2", "u3", "u4")
LENS_BENCHMARK = "EUCLID-SAGE-1k-D_VIS"
BATCH_SIZE = 50                          # images per forward pass


# ======================================================================================= 1. networks and images

def strip_quick(name):
    """A folder name without the suffix _quick of a quick run."""
    return name[:-len("_quick")] if name.endswith("_quick") else name


def find_networks(results):
    """All saved networks of a group: a list of dicts with kind, network, copy, seed, fold and path."""
    found = []
    for path in sorted((results / "simulations").glob("*/best_unet.pt")):
        copy = strip_quick(path.parent.name)
        network, init_seed = re.fullmatch(r"(.+)-init(\d+)", copy).groups()
        found.append({"kind": "simulation", "network": network, "copy": int(init_seed), "seed": 0, "fold": 0,
                      "path": path})
    # UNet-Radius-mini, with and without attention.
    for folder in results.glob("einstein_radius_real*"):
        random_weights = "random_weights" in folder.name
        for path in sorted((folder / "folds").glob("*_network.pt")):
            match = re.fullmatch(r"(pretrained-(.+)-init(\d+)|random)_seed(\d+)_fold(\d+)_network\.pt", path.name)
            if match is None:
                continue
            _, network, init_seed, seed, fold = match.groups()
            if random_weights:
                saved = torch.load(path, map_location="cpu")
                found.append({"kind": "random", "network": network_name(saved["architecture"], "radius"),
                              "copy": 0, "seed": int(seed), "fold": int(fold), "path": path})
            else:
                found.append({"kind": "finetuned", "network": network, "copy": int(init_seed), "seed": int(seed),
                              "fold": int(fold), "path": path})
    # UNet-Starlight, with and without attention.
    for path in sorted(results.glob(f"fine_tuning_1k_D*/*/{LENS_BENCHMARK}/pretrained/seed*/fold*_network.pt")):
        network, init_seed = re.fullmatch(r"(.+)-init(\d+)", strip_quick(path.parents[3].name)).groups()
        found.append({"kind": "finetuned", "network": network, "copy": int(init_seed),
                      "seed": int(path.parent.name[4:]), "fold": int(path.name[4:-len("_network.pt")]), "path": path})
    for path in sorted(results.glob(f"random_weights_1k_D*/*/{LENS_BENCHMARK}/random/seed*/fold*_network.pt")):
        found.append({"kind": "random", "network": strip_quick(path.parents[3].name), "copy": 0,
                      "seed": int(path.parent.name[4:]), "fold": int(path.name[4:-len("_network.pt")]), "path": path})
    return found


def network_name(architecture, task):
    """radius-mini or starlight, with the prefix attention- for a network with attention."""
    base = {"radius": "radius-mini", "lens": "starlight"}[task]
    return ("attention-" if architecture.get("attention") else "") + base


def network_id(entry):
    """The name of a network's RDM file: kind, network, copy (or training seed for random weights) and fold."""
    if entry["kind"] == "simulation":
        return f"simulation_{entry['network']}-init{entry['copy']:02d}"
    if entry["kind"] == "finetuned":
        return f"finetuned_{entry['network']}-init{entry['copy']:02d}_fold{entry['fold']}"
    return f"random_{entry['network']}_seed{entry['seed']:02d}_fold{entry['fold']}"


def build_network(entry):
    """The saved network as a module, and its input transform."""
    saved = torch.load(entry["path"], map_location="cpu")
    unet = UNet(**saved["architecture"])
    if entry["kind"] == "simulation":
        module = unet
    elif saved["architecture"].get("einstein_radius"):
        module = RadiusRegressor(unet)
    else:
        module = EncoderClassifier(unet)
    module.load_state_dict(saved["model_state_dict"])
    return module.eval(), saved["preprocessing"]


def simulated_test_images(simulated_path, results):
    """The test images of the split, the same for every copy (checked), and their rows in the simulated set."""
    splits = sorted((results / "simulations").glob("*/split_indices.npz"))
    test_rows = [np.load(path)["test_idx"] for path in splits]
    if any(not np.array_equal(rows, test_rows[0]) for rows in test_rows):
        raise SystemExit("The copies have different test images; they should share one split.")
    with np.load(simulated_path) as arrays:
        if len(arrays["label"]) <= test_rows[0].max():
            raise SystemExit(f"{simulated_path} has fewer images than the split of the copies expects.")
        images = arrays["images"][test_rows[0]]
    return images.astype(np.float32), test_rows[0]


def real_images(results, benchmarks, tables):
    """The real images and the fold in which each was held out: the lenses of the radius fine-tuning (in the order
    of finetune_radius.py), or the stamps of 1k-D. Returns the table and the stamps in simulation units."""
    radius_tables = sorted(results.glob("einstein_radius_real*/finetuned_*_seed1.csv"))
    if radius_tables:
        table = pd.read_csv(radius_tables[0])[["lens_id", "object_id", "grade", "measured_radius", "fold"]]
        lenses = pd.read_csv(tables / "EUCLID-SAGE-1k-R_VIS.csv")
        lenses = lenses[lenses.label == 1][["lens_id", "stamp_file", "vis_magnitude"]]
        table = table.merge(lenses, on="lens_id", how="left", validate="one_to_one")
        table["label"] = 1
        for other in radius_tables[1:]:                         # every fine-tuning used the same folds
            if not np.array_equal(pd.read_csv(other).fold.to_numpy(), table.fold.to_numpy()):
                raise SystemExit(f"{other} has other folds than {radius_tables[0]}")
    else:
        predictions = sorted(results.glob(f"*_1k_D*/*/{LENS_BENCHMARK}/*/seed*/predictions_per_stamp.csv"))
        if not predictions:
            raise SystemExit(f"No fine-tuning results found in {results}")
        table = pd.read_csv(tables / f"{LENS_BENCHMARK}.csv")
        table["fold"] = pd.read_csv(predictions[0]).fold.to_numpy()
        for other in predictions[1:]:
            if not np.array_equal(pd.read_csv(other).fold.to_numpy(), table.fold.to_numpy()):
                raise SystemExit(f"{other} has other folds than {predictions[0]}")
    table.insert(0, "row", np.arange(len(table)))
    return table, load_stamps_in_simulation_units(table, folder=benchmarks)


# ============================================================================================= 2. activations

def layer_activations(module, images, preprocessing, device, features):
    """The activations of every layer for all images (in batches), as float16 rows standardised over the units of
    each image (for layers with one or two values: the raw values, float32), the outputs and the attention maps."""
    transformed = (asinh_transform(images, float(preprocessing["asinh_scale"])) - preprocessing["mean"]) \
        / preprocessing["std"]
    collected, attention_maps = {}, []
    module.to(device)
    with torch.no_grad():
        for start in range(0, len(transformed), BATCH_SIZE):
            batch = torch.from_numpy(transformed[start:start + BATCH_SIZE, None].astype(np.float32)).to(device)
            layers, weights = run_layers(module, batch)
            attention_maps.append(weights.flatten(1).cpu().numpy())
            for name, values in layers.items():
                if values.dim() == 4 and features == "pooled":
                    values = values.mean(dim=(2, 3))
                values = values.flatten(1).float().cpu().numpy()
                if values.shape[1] > 2:
                    values = (values - values.mean(axis=1, keepdims=True)) / np.maximum(
                        values.std(axis=1, keepdims=True), 1e-12)
                    values = values.astype(np.float16)
                collected.setdefault(name, []).append(values)
    return {name: np.concatenate(parts) for name, parts in collected.items()}, np.concatenate(attention_maps)


def run_layers(module, batch):
    """Every layer of a U-Net (encoder, readout, decoder, outputs), of a RadiusRegressor or of an EncoderClassifier."""
    layers = {}
    encoder = module.encoder if hasattr(module, "encoder") else {block: getattr(module, block)
                                                                for block in ENCODER_BLOCKS}
    features, levels = batch, []
    for block in ENCODER_BLOCKS:
        features = encoder[block](features)
        layers[block] = features
        levels.append(features)
    pooled, weights = pool_bottleneck(features, module.attention, return_weights=True)
    layers["readout"] = pooled
    if isinstance(module, UNet):
        decoded = levels[-1]
        for block, skip in zip(DECODER_BLOCKS, reversed(levels[:-1])):
            decoded = getattr(module, block)(decoded, skip)
            layers[block] = decoded
        layers["out"] = module.out(decoded)
        if module.einstein_radius:
            layers["radius"] = module.radius(pooled)
        if module.lens_output:
            layers["lens_logits"] = module.lens(pooled)
    elif isinstance(module, RadiusRegressor):
        layers["radius"] = module.radius(pooled)
    else:
        layers["lens_logits"] = module.classifier(pooled)
    return layers, weights


def rdm(values):
    """Condensed RDM: 1 - Pearson correlation of standardised rows, or the Euclidean distance for 1-2 values."""
    if values.dtype != np.float16:
        differences = values[:, None, :] - values[None, :, :]
        square = np.sqrt((differences.astype(np.float64) ** 2).sum(axis=2))
    else:
        rows = torch.from_numpy(values.astype(np.float32))
        square = (1.0 - rows @ rows.T / rows.shape[1]).numpy().astype(np.float64)
    np.fill_diagonal(square, 0.0)
    return squareform((square + square.T) / 2, checks=False).astype(np.float32)


# ===================================================================================================== 3. main

def main():
    argument_parser = argparse.ArgumentParser(description="RDMs of every layer of the saved networks of one group")
    argument_parser.add_argument("--results", type=Path, required=True, help="results folder of one group of networks")
    argument_parser.add_argument("--simulated", type=Path, required=True, help="radius_training_10000.npz")
    argument_parser.add_argument("--benchmarks", type=Path, default=BENCHMARK_FOLDER,
                                 help="the folder with the 1k tables and stamps/ (default EUCLID_SAGE_BENCHMARK_DIR)")
    argument_parser.add_argument("--tables", type=Path, default=None,
                                 help="the folder with the 1k tables, if not --benchmarks")
    argument_parser.add_argument("--features", choices=["flattened", "pooled"], default="flattened",
                                 help="flattened feature maps (default) or the average of every feature map")
    argument_parser.add_argument("--kinds", nargs="+", default=["simulation", "finetuned", "random"])
    argument_parser.add_argument("--device", default="cpu", help="cpu, mps (Apple GPU) or cuda")
    argument_parser.add_argument("--out", type=Path, default=None, help="default rdms/<group folder>/")
    settings = argument_parser.parse_args()
    results = settings.results.resolve()
    out = settings.out or Path("rdms") / strip_quick(results.name)
    if settings.features == "pooled" and settings.out is None:
        out = Path(str(out) + "_pooled")
    device = torch.device(settings.device)

    networks = [entry for entry in find_networks(results) if entry["kind"] in settings.kinds]
    counts = pd.Series([entry["kind"] for entry in networks]).value_counts().to_dict()
    print(f"{len(networks)} networks in {results}: {counts}; features {settings.features}; device {device}",
          flush=True)
    simulated, simulated_rows = simulated_test_images(settings.simulated, results)
    table, stamps = real_images(results, settings.benchmarks, settings.tables or settings.benchmarks)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "real_images.csv", index=False)
    print(f"Images: {len(simulated)} simulated test images; {len(table)} real images in "
          f"{table.fold.nunique()} folds", flush=True)

    rows = []
    for number, entry in enumerate(networks, start=1):
        module, preprocessing = build_network(entry)
        identifier = network_id(entry)
        image_sets = {"simulated": (simulated, simulated_rows)}
        held_out = table.row.to_numpy() if entry["fold"] == 0 else table.row[table.fold == entry["fold"]].to_numpy()
        image_sets["real"] = (stamps[held_out], held_out)
        for image_set, (images, image_rows) in image_sets.items():
            path = out / image_set / f"{identifier}.npz"
            if not path.exists():
                start = time.time()
                activations, attention = layer_activations(module, images, preprocessing, device, settings.features)
                arrays = {f"rdm_{name}": rdm(values) for name, values in activations.items()}
                arrays["rows"] = image_rows
                outputs = [name for name in ("radius", "lens_logits") if name in activations]
                arrays.update({f"output_{name}": activations[name] for name in outputs})
                if module.attention is not None:
                    arrays["attention"] = attention.astype(np.float32)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **arrays)
                print(f"  {number}/{len(networks)} {identifier}, {image_set}: {len(images)} images, "
                      f"{len(activations)} layers, {time.time() - start:.0f} s", flush=True)
            rows.append({**{key: value for key, value in entry.items() if key != "path"}, "id": identifier,
                         "image_set": image_set, "images": len(image_rows), "attention": module.attention is not None,
                         "file": str(path.relative_to(out)), "source": str(entry["path"])})
    pd.DataFrame(rows).to_csv(out / "networks.csv", index=False)
    (out / "settings.json").write_text(json.dumps({"results": str(results), "simulated": str(settings.simulated),
                                                   "benchmarks": str(settings.benchmarks),
                                                   "features": settings.features}, indent=1))
    print(f"RDMs saved in {out}")


if __name__ == "__main__":
    main()
