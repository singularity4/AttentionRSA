"""compare_rdms.py - comparisons of the representational dissimilarity matrices (RDMs) made by compute_rdms.py

AttentionRSA.

Runs on a CPU in minutes; trains nothing.

Method
    Representational similarity analysis (RSA; Kriegeskorte et al. 2008): two representations of the same images are
    compared through their RDMs, so that layers with different numbers of units, and networks of different
    architectures, can be compared with each other and with image features. The RDMs are compared by the
    Spearman correlation of their upper triangles, which depends only on the order of the dissimilarities and not on
    their scale. The copies of a network differ only in their initial weights, so that the correlation between copies
    measures how much of a layer's representation is shared by networks trained in the same way (Mehrer et al. 2020).

Inputs
    --rdms        one or more folders made by compute_rdms.py (for example rdms/rsa_radius,
                  rdms/rsa_radius_attention, rdms/rsa_starlight). Folders whose networks were applied to the same real
                  images are compared with each other, for example UNet-Radius-mini with and without attention.
    --simulated   the simulated set (simulate.py), for the simulation parameters of the simulated images
    --parameters  optional: simulation parameters that the simulated set lacks (simulate.py --parameters-only)
    --catalogue   optional: a CSV with object_id and further values of the real galaxies (for example Sersic index,
                  size and ellipticity from the Euclid catalogue), named with --catalogue-columns

Image feature RDMs (the absolute difference of a feature between two images)
    simulated, all images     label, neighbours, galaxy_magnitude, galaxy_sersic_radius, galaxy_sersic_index,
                              galaxy_ellipticity, mass_ellipticity, external_shear (if saved, or from
                              --parameters), total_flux (log10 of the sum of the pixels)
    simulated, lenses only    einstein_radius, source_magnitude, source_sersic_radius, source_offset, arc_pixels
    real, UNet-Radius-mini    measured_radius, vis_magnitude, grade (A = 1, B = 0), catalogue values (all lenses)
    real, UNet-Starlight      label, vis_magnitude, catalogue values; lenses only: grade, vis_magnitude, catalogue

Comparisons (Spearman correlation of the upper triangles of two RDMs)
    The real images are compared fold by fold, on the stamps of one fold: a network of fold f saw none of them in
    training, and the networks trained only on simulations are restricted to the same stamps.
    layer_image_feature.csv  every layer of every network against every image feature RDM
    between_networks.csv     every layer of one network against the same layer of another, on the same images:
        copies                                          copies of one network trained in the same way (for
                                                        random weights: different training seeds)
        same copy, before and after fine-tuning         copy k trained on simulations vs copy k fine-tuned
        different copies, before and after fine-tuning
        fine-tuned vs random weights
        simulations vs random weights                   trained only on simulations vs trained only on real
                                                        images from random weights
        with vs without attention, same copy            the same training and initial weights
        with vs without attention, different copies
        Networks of different folds are not compared.
    *_summary.csv            mean, standard deviation and count (layer_image_feature: averaged over the folds first)
    report.md                the means as tables

Usage
    python compare_rdms.py --rdms rdms/rsa_radius rdms/rsa_radius_attention --simulated radius_training_10000.npz
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata

LAYER_ORDER = ["inc", "d1", "d2", "d3", "d4", "readout", "u1", "u2", "u3", "u4", "out", "radius", "lens_logits"]
SIMULATED_ALL = ["label", "neighbours", "galaxy_magnitude", "galaxy_sersic_radius", "galaxy_sersic_index",
                 "galaxy_ellipticity", "mass_ellipticity", "external_shear", "total_flux"]
SIMULATED_LENSES = ["einstein_radius", "source_magnitude", "source_sersic_radius", "source_offset", "arc_pixels"]


# ==================================================================================== 1. RDMs and image features

def load_families(folders):
    """The networks of all folders, grouped into families with the same real images."""
    families = []
    for folder in folders:
        table = pd.read_csv(folder / "real_images.csv")
        networks = pd.read_csv(folder / "networks.csv")
        networks["path"] = [str(folder / file) for file in networks.file]
        networks["group"] = folder.name
        for family in families:
            if family["table"][["stamp_file", "fold"]].equals(table[["stamp_file", "fold"]]):
                family["networks"] = pd.concat([family["networks"], networks], ignore_index=True)
                family["groups"].append(folder.name)
                break
        else:
            families.append({"table": table, "networks": networks, "groups": [folder.name]})
    for family in families:
        networks = family["networks"]
        networks["instance"] = np.where(networks["kind"] == "random", networks["seed"], networks["copy"])
        family["task"] = "radius" if "measured_radius" in family["table"] else "lens"
    return families


def rdm_of(path, layer):
    """The condensed RDM of one layer, read from an RDM file of compute_rdms.py, or None if the network lacks the
    layer."""
    with np.load(path) as arrays:
        return arrays[f"rdm_{layer}"] if f"rdm_{layer}" in arrays.files else None


def rows_of(path):
    """The rows of the images of an RDM file: test indices of the simulated set, or rows of the real table."""
    with np.load(path) as arrays:
        return arrays["rows"]


def layers_of(path):
    """The layers whose RDMs an RDM file holds."""
    with np.load(path) as arrays:
        return [name[4:] for name in arrays.files if name.startswith("rdm_")]


def ranks(condensed, positions=None):
    """Standardised ranks of the upper triangle of an RDM, restricted to the images at positions: the dot product of
    two such vectors divided by their length is Spearman's rho."""
    if positions is not None:
        condensed = squareform(squareform(condensed)[np.ix_(positions, positions)], checks=False)
    values = rankdata(condensed)
    spread = values.std()
    return ((values - values.mean()) / spread if spread > 0 else np.zeros_like(values)).astype(np.float32)


def simulated_image_features(simulated_path, rows, parameters_path=None):
    """The image features of the simulated test images. Simulation parameters that the simulated set lacks
    (mass_ellipticity and external_shear in sets made before they were saved) are read from parameters_path, made
    with simulate.py --parameters-only and the same seed."""
    with np.load(simulated_path) as arrays:
        values = {name: arrays[name][rows].astype(np.float64) for name in
                  ["label", "neighbours", "galaxy_magnitude", "galaxy_sersic_radius", "galaxy_sersic_index",
                   "galaxy_ellipticity", "mass_ellipticity", "external_shear", "source_magnitude",
                   "source_sersic_radius", "source_offset"] if name in arrays.files}
        values["einstein_radius"] = arrays["theta_E"][rows].astype(np.float64)
        values["total_flux"] = np.log10(np.clip(arrays["images"][rows].sum(axis=(1, 2)), 1e-6, None))
        values["arc_pixels"] = arrays["mask"][rows].reshape(len(rows), -1).sum(axis=1).astype(np.float64)
        labels = arrays["label"]
    if parameters_path is not None:
        with np.load(parameters_path) as parameters:
            if not np.array_equal(parameters["label"], labels):
                raise SystemExit(f"{parameters_path} does not belong to {simulated_path} (different labels)")
            for name in ("mass_ellipticity", "external_shear"):
                if name not in values and name in parameters.files:
                    values[name] = parameters[name][rows].astype(np.float64)
    return pd.DataFrame(values)


def real_image_features(table, catalogue, catalogue_columns):
    """The image features of the real images: label, VIS magnitude, grade (A = 1, B = 0), the measured Einstein
    radius (UNet-Radius-mini) and the catalogue values."""
    values = pd.DataFrame({"label": table.label.astype(float), "vis_magnitude": table.vis_magnitude.astype(float)})
    values["grade"] = np.where(table.grade == "A", 1.0, np.where(table.grade == "B", 0.0, np.nan))
    if "measured_radius" in table:
        values["measured_radius"] = table.measured_radius.astype(float)
    if catalogue is not None:
        merged = table[["object_id"]].merge(catalogue[["object_id", *catalogue_columns]], on="object_id", how="left")
        for column in catalogue_columns:
            values[column] = merged[column].astype(float).to_numpy()
    return values


def subsets(image_set, image_features, task):
    """(subset name, image mask, image feature names) of an image set."""
    lenses = image_features.label.to_numpy() == 1
    everything = np.ones(len(image_features), bool)
    if image_set == "simulated":
        return [("all", everything, [name for name in SIMULATED_ALL if name in image_features]),
                ("lenses", lenses, SIMULATED_LENSES)]
    others = [name for name in image_features.columns if name not in ("label", "grade")]
    if task == "radius":                                    # UNet-Radius-mini: all real images are lenses
        return [("lenses", everything, others + ["grade"])]
    return [("all", everything, ["label"] + others), ("lenses", lenses, ["grade"] + others)]


# ========================================================================================== 2. comparisons

def blocks(family, image_set):
    """The sets of images on which networks are compared: all simulated test images, or the real stamps of one
    fold. Yields (fold, the networks with their positions in their own rows, the rows of the images)."""
    networks = family["networks"][family["networks"].image_set == image_set].reset_index(drop=True)
    if image_set == "simulated":
        rows = rows_of(networks.path.iloc[0])
        for path in networks.path:
            if not np.array_equal(rows_of(path), rows):
                raise SystemExit(f"{path} has other simulated images than {networks.path.iloc[0]}")
        yield 0, networks.assign(positions=[None] * len(networks)), rows
        return
    fold_of = family["table"].fold.to_numpy()
    for fold in sorted(np.unique(fold_of)):
        rows = np.flatnonzero(fold_of == fold)
        members = networks[(networks.fold == 0) | (networks.fold == fold)].reset_index(drop=True)
        positions = []
        for path in members.path:
            own = rows_of(path)
            found = np.flatnonzero(np.isin(own, rows))
            if len(found) != len(rows):
                raise SystemExit(f"{path} lacks stamps of fold {fold}")
            positions.append(None if len(found) == len(own) else found)
        yield fold, members.assign(positions=positions), rows


def pair_type(first, second):
    """The kind of comparison of two networks (rows of networks.csv), or None if they are not compared. Networks of
    different folds were fine-tuned on different stamps and are not compared."""
    if first.network.removeprefix("attention-") != second.network.removeprefix("attention-"):
        return None
    if first.fold and second.fold and first.fold != second.fold:
        return None
    if first.network.startswith("attention-") != second.network.startswith("attention-"):
        if first.kind != second.kind:
            return None
        same_copy = first.instance == second.instance
        return f"with vs without attention, {'same copy' if same_copy else 'different copies'}"
    kinds = {first.kind, second.kind}
    if first.kind == second.kind:
        return "copies" if first.instance != second.instance else None
    if kinds == {"simulation", "finetuned"}:
        return ("same copy, before and after fine-tuning" if first.instance == second.instance
                else "different copies, before and after fine-tuning")
    if kinds == {"finetuned", "random"}:
        return "fine-tuned vs random weights"
    if kinds == {"simulation", "random"}:
        return "simulations vs random weights"
    return None


def compare(families, simulated, catalogue, catalogue_columns, parameters=None):
    """Both comparisons, block by block and layer by layer; all correlations of one layer come from one matrix
    product of the standardised ranks."""
    feature_rows, pair_rows = [], []
    for family in families:
        for image_set in ("simulated", "real"):
            if not (family["networks"].image_set == image_set).any():
                continue
            for fold, members, rows in blocks(family, image_set):
                image_features = (simulated_image_features(simulated, rows, parameters) if image_set == "simulated"
                                  else real_image_features(family["table"].iloc[rows].reset_index(drop=True),
                                                           catalogue, catalogue_columns))
                layers = sorted({layer for path in members.path for layer in layers_of(path)},
                                key=lambda name: LAYER_ORDER.index(name) if name in LAYER_ORDER else 99)
                pairs = [(first, second, pair_type(members.iloc[first], members.iloc[second]))
                         for first in range(len(members)) for second in range(first + 1, len(members))]
                pairs = [pair for pair in pairs if pair[2] is not None]
                for layer in layers:
                    condensed = {member: rdm_of(path, layer) for member, path in enumerate(members.path)}
                    present = [member for member in condensed if condensed[member] is not None]
                    # Networks against networks: one row of standardised ranks per network.
                    all_image_ranks = np.stack([ranks(condensed[member], members.positions[member])
                                                for member in present])
                    correlations = all_image_ranks @ all_image_ranks.T / all_image_ranks.shape[1]
                    row_of = {member: row for row, member in enumerate(present)}
                    for first, second, kind in pairs:
                        if first in row_of and second in row_of:
                            network_a, network_b = members.iloc[first], members.iloc[second]
                            pair_rows.append({"pair": kind, "image_set": image_set, "fold": fold, "layer": layer,
                                              "kind_a": network_a.kind, "network_a": network_a.network,
                                              "instance_a": network_a.instance,
                                              "kind_b": network_b.kind, "network_b": network_b.network,
                                              "instance_b": network_b.instance,
                                              "rho": float(correlations[row_of[first], row_of[second]])})
                    # Layers against image features.
                    restricted = {}                 # standardised ranks of the layer for each set of kept images
                    for subset, mask, names in subsets(image_set, image_features, family["task"]):
                        for name in names:
                            values = image_features[name].to_numpy()
                            keep = mask & np.isfinite(values)
                            if keep.sum() < 4:
                                continue
                            target = ranks(pdist(values[keep, None], "cityblock"))
                            if keep.all():
                                layer_ranks = all_image_ranks
                            else:
                                if keep.tobytes() not in restricted:
                                    restricted[keep.tobytes()] = np.stack(
                                        [ranks(condensed[member], compose(members.positions[member], keep))
                                         for member in present])
                                layer_ranks = restricted[keep.tobytes()]
                            feature_correlations = layer_ranks @ target / len(target)
                            for member, rho in zip(present, feature_correlations):
                                network = members.iloc[member]
                                feature_rows.append({"group": network.group, "kind": network.kind,
                                                      "network": network.network, "instance": network.instance,
                                                      "fold": fold, "image_set": image_set, "subset": subset,
                                                      "layer": layer, "image_feature": name, "rho": float(rho),
                                                      "images": int(keep.sum())})
    return pd.DataFrame(feature_rows), pd.DataFrame(pair_rows)


def compose(positions, keep):
    """Positions of the kept images of a block in a network's own rows."""
    selected = np.flatnonzero(keep)
    return selected if positions is None else positions[selected]


# ========================================================================================== 3. summaries

def ordered(table):
    """The table with the layers as an ordered category, from the input to the outputs."""
    table = table.copy()
    table["layer"] = pd.Categorical(table.layer, [name for name in LAYER_ORDER if name in set(table.layer)],
                                    ordered=True)
    return table


def summaries(feature_table, pair_table):
    """Mean, standard deviation and count of the correlations. The correlation of a layer with an image feature is
    first averaged over the folds of each network, so that each network counts once."""
    keys = ["group", "kind", "network", "image_set", "subset", "layer", "image_feature"]
    per_instance = feature_table.groupby(keys + ["instance"]).rho.mean().reset_index()
    feature_summary = ordered(per_instance.groupby(keys).rho.agg(["mean", "std", "count"]).reset_index())
    if pair_table.empty:
        return feature_summary.sort_values(keys), pair_table
    pair_keys = ["pair", "kind_a", "network_a", "kind_b", "network_b", "image_set", "layer"]
    pair_summary = ordered(pair_table.groupby(pair_keys).rho.agg(["mean", "std", "count"]).reset_index())
    return feature_summary.sort_values(keys), pair_summary.sort_values(pair_keys)


def markdown(table):
    """A table in Markdown, with two decimals."""
    lines = ["| " + " | ".join(map(str, table.columns)) + " |", "|" + "---|" * len(table.columns)]
    for _, row in table.iterrows():
        lines.append("| " + " | ".join(f"{value:.2f}" if isinstance(value, float) else str(value)
                                       for value in row) + " |")
    return "\n".join(lines)


def report(feature_summary, pair_summary, out):
    """report.md: the mean correlations, one table per network and image set, and one per kind of network pair."""
    lines = ["# RSA comparisons", "",
             "Spearman correlation between RDMs, mean over copies; standard deviations and counts in the CSV files.",
             "", "## Layers against image features", ""]
    for (kind, network, image_set, subset), part in feature_summary.groupby(
            ["kind", "network", "image_set", "subset"], observed=True):
        wide = part.pivot_table(index="layer", columns="image_feature", values="mean", observed=True).reset_index()
        lines += [f"### {network}, {kind}: {image_set} images, {subset}", "", markdown(wide), ""]
    lines += ["## Networks against networks, layer by layer", ""]
    if not pair_summary.empty:
        labelled = pair_summary.assign(comparison=pair_summary.pair + ": " + pair_summary.network_a + " "
                                       + pair_summary.kind_a + " / " + pair_summary.network_b + " "
                                       + pair_summary.kind_b)
        for (image_set, pair), part in labelled.groupby(["image_set", "pair"], observed=True):
            wide = part.pivot_table(index="layer", columns="comparison", values="mean", observed=True).reset_index()
            wide.columns = [column.split(": ", 1)[-1] for column in wide.columns]
            lines += [f"### {pair}: {image_set} images", "", markdown(wide), ""]
    (out / "report.md").write_text("\n".join(lines) + "\n")


def main():
    argument_parser = argparse.ArgumentParser(description="Comparisons of the RDMs of compute_rdms.py")
    argument_parser.add_argument("--rdms", type=Path, nargs="+", required=True, help="folders made by compute_rdms.py")
    argument_parser.add_argument("--simulated", type=Path, required=True, help="the simulated set (simulate.py)")
    argument_parser.add_argument("--parameters", type=Path, default=None,
                                 help="simulation parameters missing from --simulated (simulate.py --parameters-only)")
    argument_parser.add_argument("--catalogue", type=Path, default=None,
                                 help="CSV with object_id and catalogue values")
    argument_parser.add_argument("--catalogue-columns", nargs="*", default=[], help="columns of --catalogue to use")
    argument_parser.add_argument("--out", type=Path, default=None, help="default comparisons/<rdms folder names>/")
    settings = argument_parser.parse_args()
    folders = [folder.resolve() for folder in settings.rdms]
    out = settings.out or Path("comparisons") / "+".join(folder.name for folder in folders)
    out.mkdir(parents=True, exist_ok=True)
    catalogue = pd.read_csv(settings.catalogue) if settings.catalogue else None

    families = load_families(folders)
    print(f"{sum(len(family['networks']) for family in families)} RDM files in {len(families)} "
          f"group{'s' if len(families) > 1 else ''} of real images", flush=True)
    feature_table, pair_table = compare(families, settings.simulated, catalogue, settings.catalogue_columns,
                                         settings.parameters)
    feature_table.to_csv(out / "layer_image_feature.csv", index=False)
    pair_table.to_csv(out / "between_networks.csv", index=False)
    print(f"{len(feature_table)} layer-image feature and {len(pair_table)} network-network correlations", flush=True)
    feature_summary, pair_summary = summaries(feature_table, pair_table)
    feature_summary.to_csv(out / "layer_image_feature_summary.csv", index=False)
    pair_summary.to_csv(out / "between_networks_summary.csv", index=False)
    report(feature_summary, pair_summary, out)
    print(f"Saved in {out}")


if __name__ == "__main__":
    main()
