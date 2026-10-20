"""train_copies.py - training of the U-Nets on simulated images: copies of a network that differ only in their initial
weights

AttentionRSA.

Task
    For every pixel of a 101 x 101 stamp the network gives the probability that the pixel contains light of a lensed
    source. UNet-Radius-mini also estimates the Einstein radius of the stamp; UNet-Starlight with --lens-output also
    decides whether the stamp contains a lens. With --attention, this second output is read from the bottleneck
    through an attention map (models.py).

Copies (--init-seed)
    With --init-seed k the initial weights are drawn with the seed k, while the split, the order of the images, the
    augmentation and the dropout follow --seed and are the same for every copy. Copies with the same --init-seed,
    with and without --attention, start from the same weights in every layer they share. Without --init-seed the
    initial weights also follow --seed. Since everything else is the same, the differences between copies come from
    the initial weights alone (as in Mehrer et al. 2020).

Data
    A simulated set made by simulate.py (images, arc masks, labels and Einstein radii). The input transform, the
    augmentation and the stratified 80 / 10 / 10 split into training, validation and test images are described in
    data.py.

Loss
    Binary cross-entropy of the pixel logits plus the soft Dice loss:
        L_seg = BCE_w + (1 - mean over images of (2 sum p y + eps) / (sum p + sum y + eps)),
    with arc probabilities p, true mask y and eps = 1e-6. Arc pixels are rare, so the cross-entropy of arc pixels is
    weighted by w = sqrt(N_background / N_arc), computed from the training masks and limited to [1, 10]. The Dice term
    is computed per image, so that arc pixels predicted in a non-lens, whose mask is empty, increase the loss.
    UNet-Radius-mini: plus lambda x the mean absolute error of the estimated Einstein radius over the lenses of the
    batch (non-lenses have no Einstein radius), lambda = 1 (--einstein-radius-weight), radii in arcseconds.
    --lens-output: plus the cross-entropy of the lens label, weighted by --lens-weight (1).

Optimisation and choice of the epoch
    AdamW with learning rate 3e-4 and weight decay 1e-4, batches of 16 images, gradient norm limited to 5. The
    learning rate is halved when the validation criterion has not improved for more than 8 epochs, and training stops
    after 25 epochs without improvement or after 150 epochs. The criterion is the mean Dice coefficient of the
    validation lenses; the weights of the best epoch are kept, and the test images are evaluated once with them.

Outputs (--out, default results/simulations/<network>-init<k>/; with _quick for a quick run)
    best_unet.pt          weights of the best epoch, architecture, input transform and decision rule
    history.csv           training loss and validation metrics of every epoch
    split_indices.npz     indices of the training, validation and test images
    test_predictions.npz  arc probabilities of the test images (UNet-Radius-mini: also true and estimated radii)
    summary.json          settings and test results
    test_examples.png     test images with true and predicted arc masks, for visual inspection

Usage
    python train_copies.py --data radius_training_10000.npz --model radius-mini --init-seed 1
    python train_copies.py --data radius_training_10000.npz --model radius-mini --attention --init-seed 1
    python train_copies.py --data radius_training_10000.npz --model starlight --lens-output --attention --init-seed 1
    python train_copies.py --data radius_training_200.npz --model radius-mini --init-seed 1 --quick    # 5 epochs
"""

import argparse
import csv
import json
import math
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import (RESULTS_FOLDER, ArcSegmentationDataset, IntensityTransform, check_simulated_set,
                  read_simulated_set, seed_data_loader_worker, set_random_seeds, split_simulated_set)
from metrics import einstein_radius_metrics, mean_lens_dice, segmentation_metrics
from models import EINSTEIN_RADIUS_MODELS, FILTERS, NAMES, architecture_of, build_unet, choose_device


# =========================================================================================================== 1. loss

def soft_dice_loss(logits, target, eps=1e-6):
    """1 - mean over images of (2 sum p y + eps) / (sum p + sum y + eps), with p the arc probabilities."""
    probabilities = torch.sigmoid(logits)
    pixel_dimensions = (1, 2, 3)
    overlap = (probabilities * target).sum(pixel_dimensions)
    total = probabilities.sum(pixel_dimensions) + target.sum(pixel_dimensions)
    dice = (2.0 * overlap + eps) / (total + eps)
    return 1.0 - dice.mean()


class CrossEntropyDiceLoss(nn.Module):
    """Binary cross-entropy with the weight w for arc pixels, plus the soft Dice loss."""

    def __init__(self, positive_class_weight, dice_weight=1.0):
        super().__init__()
        self.register_buffer("positive_class_weight",
                             torch.tensor([float(positive_class_weight)], dtype=torch.float32))
        self.dice_weight = float(dice_weight)

    def forward(self, logits, target):
        cross_entropy = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.positive_class_weight)
        return cross_entropy + self.dice_weight * soft_dice_loss(logits, target)


def arc_pixel_weight(training_masks):
    """w = sqrt(N_background / N_arc) of the training masks, limited to [1, 10]."""
    arc_pixels = float(training_masks.sum())
    background_pixels = float(training_masks.size) - arc_pixels
    return float(np.clip(math.sqrt(background_pixels / max(arc_pixels, 1.0)), 1.0, 10.0))


# ============================================================================================= 2. prediction

@torch.no_grad()
def predict_arc_probabilities(model, loader, device, estimated_radii=None):
    """Arc probabilities of all images of a loader. Returns (probabilities, true masks, labels, indices). If
    estimated_radii is a list, the estimated Einstein radii (UNet-Radius-mini) are appended to it."""
    model.eval()
    probabilities, true_masks, labels, indices = [], [], [], []
    for image_batch, mask_batch, label_batch, index_batch in loader:
        image_batch = image_batch.to(device, non_blocking=True)
        if estimated_radii is None:
            logits = model(image_batch)
        else:
            logits, radii = model(image_batch, return_einstein_radius=True)
            estimated_radii.extend(radii.cpu().numpy().tolist())
        probabilities.append(torch.sigmoid(logits).cpu().numpy()[:, 0])
        true_masks.append(mask_batch.numpy()[:, 0])
        labels.append(label_batch.numpy())
        indices.append(index_batch.numpy())
    return np.concatenate(probabilities), np.concatenate(true_masks), np.concatenate(labels), np.concatenate(indices)


def save_example_figure(path, raw_images, probabilities, true_masks, labels, indices, threshold, max_examples=12):
    """One row per test image, lenses first: image, true arc mask, arc probability and predicted arc mask."""
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    shown = (list(np.where(labels == 1)[0]) + list(np.where(labels == 0)[0]))[:max_examples]
    if not shown:
        return False
    figure, axes = plt.subplots(len(shown), 4, figsize=(9, 2.2 * len(shown)))
    if len(shown) == 1:
        axes = axes[None, :]
    for row, example in enumerate(shown):
        image_index = int(indices[example])
        image = raw_images[image_index]
        lower, upper = np.percentile(image, [1, 99.5])
        axes[row, 0].imshow(np.clip(image, lower, upper), cmap="gray")
        axes[row, 0].set_title(f"image {image_index}, label {labels[example]}")
        axes[row, 1].imshow(true_masks[example], cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("true arc mask")
        axes[row, 2].imshow(probabilities[example], cmap="magma", vmin=0, vmax=1)
        axes[row, 2].set_title("arc probability")
        axes[row, 3].imshow(probabilities[example] >= threshold, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].set_title(f"probability >= {threshold:.2f}")
        for column in range(4):
            axes[row, column].axis("off")
    plt.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return True


# =============================================================================================== 3. training

def network_name(settings):
    """radius-mini, attention-radius-mini, starlight or attention-starlight."""
    return ("attention-" if settings.attention else "") + settings.model


def parse_arguments():
    argument_parser = argparse.ArgumentParser(description="Training of the U-Nets on simulated images")
    argument_parser.add_argument("--data", type=Path, required=True, help="simulated set (.npz, or .zip with one .npz)")
    argument_parser.add_argument("--model", choices=list(FILTERS), required=True,
                                 help="starlight (16 filters in the first block) or radius-mini (8 filters, with the "
                                      "Einstein radius output)")
    argument_parser.add_argument("--attention", action="store_true",
                                 help="read the radius or lens output from the bottleneck through an attention map")
    argument_parser.add_argument("--lens-output", action="store_true",
                                 help="also train a lens output (non-lens, lens) on the labels")
    argument_parser.add_argument("--init-seed", type=int, default=None,
                                 help="seed of the initial weights only (copy k: --init-seed k); default: --seed")
    argument_parser.add_argument("--einstein-radius-weight", type=float, default=1.0,
                                 help="lambda, the weight of the Einstein radius error in the loss (radius-mini)")
    argument_parser.add_argument("--lens-weight", type=float, default=1.0,
                                 help="weight of the cross-entropy of the lens label in the loss (--lens-output)")
    argument_parser.add_argument("--out", type=Path, default=None,
                                 help="output folder (default results/simulations/<network>-init<k>)")
    argument_parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    argument_parser.add_argument("--epochs", type=int, default=150, help="largest number of epochs")
    argument_parser.add_argument("--patience", type=int, default=25,
                                 help="training stops after this many epochs without improvement")
    argument_parser.add_argument("--batch-size", type=int, default=16)
    argument_parser.add_argument("--learning-rate", type=float, default=3e-4)
    argument_parser.add_argument("--weight-decay", type=float, default=1e-4)
    argument_parser.add_argument("--dropout", type=float, default=0.10)
    argument_parser.add_argument("--asinh-scale", type=float, default=0.02, help="scale s of the input transform")
    argument_parser.add_argument("--threshold", type=float, default=0.50, help="arc probability threshold")
    argument_parser.add_argument("--min-pixels", type=int, default=40,
                                 help="smallest number of arc pixels for a stamp to be classified as a lens")
    argument_parser.add_argument("--seed", type=int, default=20260925, help="seed of the split and of the training")
    argument_parser.add_argument("--workers", type=int, default=0,
                                 help="processes that load images in parallel; 0 loads them in the main process")
    argument_parser.add_argument("--quick", action="store_true", help="5 epochs, to test the setup")
    settings = argument_parser.parse_args()
    if settings.out is None:
        copy = network_name(settings) + ("" if settings.init_seed is None else f"-init{settings.init_seed:02d}")
        settings.out = RESULTS_FOLDER / "simulations" / (copy + ("_quick" if settings.quick else ""))
    return settings


def main():
    settings = parse_arguments()
    set_random_seeds(settings.seed)
    settings.out.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    arrays = read_simulated_set(settings.data)
    images, arc_masks, labels = check_simulated_set(arrays)
    number_of_images, height, width = images.shape
    with_einstein_radius = settings.model in EINSTEIN_RADIUS_MODELS
    if with_einstein_radius:
        if "theta_E" not in arrays:
            raise SystemExit(f"{settings.data} contains no Einstein radii (theta_E), which {NAMES[settings.model]} "
                             "needs")
        einstein_radii = np.asarray(arrays["theta_E"], dtype=np.float32)
        einstein_radii_tensor = torch.from_numpy(einstein_radii)
    training_indices, validation_indices, test_indices = split_simulated_set(labels, settings.seed)
    transform = IntensityTransform.fit(images[training_indices], scale=settings.asinh_scale)

    training_set = ArcSegmentationDataset(images, arc_masks, labels, training_indices, transform, augment=True)
    validation_set = ArcSegmentationDataset(images, arc_masks, labels, validation_indices, transform, augment=False)
    test_set = ArcSegmentationDataset(images, arc_masks, labels, test_indices, transform, augment=False)
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(settings.seed)
    loader_settings = dict(batch_size=settings.batch_size, num_workers=settings.workers,
                           pin_memory=torch.cuda.is_available(),
                           worker_init_fn=seed_data_loader_worker if settings.workers > 0 else None)
    training_loader = DataLoader(training_set, shuffle=True, generator=shuffle_generator, **loader_settings)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_settings)
    test_loader = DataLoader(test_set, shuffle=False, **loader_settings)

    device = choose_device(settings.device)
    if settings.init_seed is None:
        model = build_unet(settings.model, dropout=settings.dropout, attention=settings.attention,
                           lens_output=settings.lens_output).to(device)
    else:
        # The initial weights are drawn with --init-seed. fork_rng restores the random generators afterwards (CPU and
        # GPUs), so that the dropout and everything else still follow --seed and are the same for every copy.
        gpus = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=gpus):
            torch.manual_seed(settings.init_seed)
            model = build_unet(settings.model, dropout=settings.dropout, attention=settings.attention,
                               lens_output=settings.lens_output)
        model = model.to(device)
    filters = FILTERS[settings.model]
    number_of_parameters = sum(parameter.numel() for parameter in model.parameters())
    positive_class_weight = arc_pixel_weight(arc_masks[training_indices])
    loss_function = CrossEntropyDiceLoss(positive_class_weight=positive_class_weight, dice_weight=1.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    learning_rate_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                                                        patience=8, min_lr=1e-6)
    maximum_epochs = 5 if settings.quick else settings.epochs
    patience = 5 if settings.quick else settings.patience

    print(f"{'Attention ' if settings.attention else ''}{NAMES[settings.model]}: training on simulated images\n"
          f"  data          {settings.data}\n"
          f"  images        {number_of_images} of {height} x {width} pixels, {int((labels == 1).sum())} lenses, "
          f"{int((labels == 0).sum())} non-lenses\n"
          f"  split         {len(training_indices)} training, {len(validation_indices)} validation, "
          f"{len(test_indices)} test images\n"
          f"  device        {device}\n"
          f"  parameters    {number_of_parameters:,}; filters "
          f"{', '.join(str(filters * 2 ** level) for level in range(5))}\n"
          f"  loss          cross-entropy (arc pixel weight {positive_class_weight:.3f}) + soft Dice"
          + (f" + {settings.einstein_radius_weight:g} x Einstein radius error" if with_einstein_radius else "")
          + (f" + {settings.lens_weight:g} x lens cross-entropy" if settings.lens_output else "") + "\n"
          f"  readout       {'attention-weighted' if settings.attention else 'plain'} average of the bottleneck"
          + (f"; initial weights from --init-seed {settings.init_seed}" if settings.init_seed is not None else "") + "\n"
          f"  optimisation  AdamW, learning rate {settings.learning_rate:g}, weight decay {settings.weight_decay:g}, "
          f"batches of {settings.batch_size}, at most {maximum_epochs} epochs, patience {patience}\n"
          f"  results       {settings.out}")

    history = []
    best_state, best_epoch, best_validation_dice = None, -1, -np.inf
    epochs_without_improvement = 0
    for epoch in range(1, maximum_epochs + 1):
        model.train()
        summed_loss, images_seen = 0.0, 0
        for image_batch, mask_batch, label_batch, index_batch in training_loader:
            image_batch = image_batch.to(device, non_blocking=True)
            mask_batch = mask_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if with_einstein_radius or settings.lens_output:
                outputs = model(image_batch, return_einstein_radius=with_einstein_radius,
                                return_lens_logits=settings.lens_output)
                logits = outputs[0]
                loss = loss_function(logits, mask_batch)
                if with_einstein_radius:
                    estimated_radii = outputs[1]
                    is_lens = (label_batch == 1).to(device)
                    if is_lens.any():                       # non-lenses have no Einstein radius
                        true_radii = einstein_radii_tensor[index_batch].to(device)
                        radius_error = (estimated_radii[is_lens] - true_radii[is_lens]).abs().mean()
                        loss = loss + settings.einstein_radius_weight * radius_error
                if settings.lens_output:                    # cross-entropy of the lens label
                    loss = loss + settings.lens_weight * F.cross_entropy(outputs[-1], label_batch.to(device))
            else:
                logits = model(image_batch)
                loss = loss_function(logits, mask_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            summed_loss += float(loss.item()) * len(image_batch)
            images_seen += len(image_batch)

        training_loss = summed_loss / max(images_seen, 1)
        validation_radii = [] if with_einstein_radius else None
        validation_probabilities, validation_masks, validation_labels, validation_image_indices = \
            predict_arc_probabilities(model, validation_loader, device, validation_radii)
        validation_dice = mean_lens_dice(validation_probabilities, validation_masks, validation_labels,
                                         threshold=settings.threshold)
        validation_metrics = segmentation_metrics(validation_probabilities, validation_masks, validation_labels,
                                                  settings.threshold, settings.min_pixels)
        learning_rate_schedule.step(validation_dice)
        learning_rate = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch,
                        "train_loss": training_loss,
                        "val_lens_dice": validation_dice,
                        "val_lens_iou": validation_metrics["lens_iou_mean"],
                        "val_nonlens_zero_fraction": validation_metrics["nonlens_zero_mask_fraction"],
                        "val_mask_balanced_accuracy": validation_metrics["mask_based_lens_balanced_accuracy"],
                        "lr": learning_rate})
        if with_einstein_radius:
            is_lens = validation_labels == 1
            history[-1]["val_einstein_radius_mae"] = float(np.mean(np.abs(
                np.array(validation_radii)[is_lens] - einstein_radii[validation_image_indices][is_lens])))

        improved = validation_dice > best_validation_dice + 1e-5
        if improved:
            best_validation_dice, best_epoch = validation_dice, epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch == 1 or epoch % 5 == 0 or improved or epochs_without_improvement >= patience:
            print(f"epoch {epoch:3d}  training loss {training_loss:.4f}  validation lens Dice {validation_dice:.4f}, "
                  f"IoU {validation_metrics['lens_iou_mean']:.4f}, empty non-lens masks "
                  f"{validation_metrics['nonlens_zero_mask_fraction']:.3f}  learning rate {learning_rate:.2e}  "
                  f"best {best_validation_dice:.4f} (epoch {best_epoch})"
                  + (f"  Einstein radius error {history[-1]['val_einstein_radius_mae']:.3f}\""
                     if with_einstein_radius else ""))
        if epochs_without_improvement >= patience:
            print(f"Training stopped: no improvement of the validation lens Dice for {patience} epochs.")
            break
    if best_state is None:
        raise RuntimeError("No epoch improved the validation lens Dice")

    # ------------------------------------------------------------------ test images, evaluated once
    model.load_state_dict(best_state)
    test_radii = [] if with_einstein_radius else None
    test_probabilities, test_masks, test_labels, test_image_indices = predict_arc_probabilities(
        model, test_loader, device, test_radii)
    test_metrics = segmentation_metrics(test_probabilities, test_masks, test_labels, settings.threshold,
                                        settings.min_pixels)
    (true_nonlenses, false_alarms), (missed_lenses, found_lenses) = test_metrics["mask_based_lens_confusion_0_1"]
    print(f"\nTest images, best epoch {best_epoch} (validation lens Dice {best_validation_dice:.4f})\n"
          f"  lens Dice, mean / median       {test_metrics['lens_dice_mean']:.4f} / "
          f"{test_metrics['lens_dice_median']:.4f}\n"
          f"  lens IoU, mean / median        {test_metrics['lens_iou_mean']:.4f} / "
          f"{test_metrics['lens_iou_median']:.4f}\n"
          f"  non-lenses with an empty mask  {test_metrics['nonlens_zero_mask_fraction']:.4f}\n"
          f"  lens decision                  {found_lenses} of {found_lenses + missed_lenses} lenses found, "
          f"{false_alarms} false alarms among {true_nonlenses + false_alarms} non-lenses")
    radius_metrics = None
    if with_einstein_radius:
        radius_metrics = einstein_radius_metrics(einstein_radii[test_image_indices], test_radii, test_labels)
        print(f"  Einstein radius                mean absolute error {radius_metrics['mean_absolute_error']:.3f}\", "
              f"median {radius_metrics['median_absolute_error']:.3f}\", "
              f"mean error {radius_metrics['mean_error']:+.3f}\"")

    # ------------------------------------------------------------------ outputs
    history_path = settings.out / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    source_index = arrays.get("source_index", np.arange(number_of_images))
    split_path = settings.out / "split_indices.npz"
    np.savez_compressed(split_path, train_idx=training_indices, val_idx=validation_indices, test_idx=test_indices,
                        train_source_index=source_index[training_indices],
                        val_source_index=source_index[validation_indices],
                        test_source_index=source_index[test_indices])

    predictions_path = settings.out / "test_predictions.npz"
    radius_arrays = ({"theta_E": einstein_radii[test_image_indices],
                      "estimated_theta_E": np.array(test_radii, dtype=np.float32)} if with_einstein_radius else {})
    np.savez_compressed(predictions_path, probability=test_probabilities.astype(np.float32),
                        truth_mask=test_masks.astype(np.uint8), label=test_labels.astype(np.int64),
                        dataset_index=test_image_indices.astype(np.int64),
                        source_index=source_index[test_image_indices], **radius_arrays)

    network_path = settings.out / "best_unet.pt"
    torch.save({"model_state_dict": best_state,
                "architecture": architecture_of(settings.model, settings.dropout, attention=settings.attention,
                                                lens_output=settings.lens_output),
                "einstein_radius_weight": settings.einstein_radius_weight if with_einstein_radius else None,
                "preprocessing": transform.as_dict(),
                "threshold": settings.threshold,
                "min_pixels": settings.min_pixels,
                "best_epoch": best_epoch,
                "best_val_lens_dice": best_validation_dice,
                "seed": settings.seed,
                "init_seed": settings.init_seed}, network_path)

    example_path = settings.out / "test_examples.png"
    made_figure = save_example_figure(example_path, images, test_probabilities, test_masks, test_labels,
                                      test_image_indices, settings.threshold)

    summary = {"model": settings.model,
               "network": network_name(settings),
               "data": str(settings.data),
               "seed": settings.seed,
               "init_seed": settings.init_seed,
               "attention": settings.attention,
               "lens_output": settings.lens_output,
               "n_images": int(number_of_images),
               "image_shape": [int(height), int(width)],
               "n_lens": int((labels == 1).sum()),
               "n_nonlens": int((labels == 0).sum()),
               "split_sizes": {"train": len(training_indices), "val": len(validation_indices),
                               "test": len(test_indices)},
               "device": str(device),
               "model_parameters": int(number_of_parameters),
               "training": {"filters_first_block": filters,
                            "dropout": settings.dropout,
                            "batch_size": settings.batch_size,
                            "lr": settings.learning_rate,
                            "weight_decay": settings.weight_decay,
                            "pos_weight": positive_class_weight,
                            "einstein_radius_weight": settings.einstein_radius_weight if with_einstein_radius else None,
                            "lens_weight": settings.lens_weight if settings.lens_output else None,
                            "max_epochs": maximum_epochs,
                            "patience": patience,
                            "best_epoch": best_epoch,
                            "best_val_lens_dice": best_validation_dice},
               "preprocessing": transform.as_dict(),
               "test_at_requested_threshold": test_metrics,
               "einstein_radius": radius_metrics,
               "runtime_seconds": time.time() - start_time,
               "outputs": {"checkpoint": str(network_path),
                           "history": str(history_path),
                           "split_indices": str(split_path),
                           "test_predictions": str(predictions_path),
                           "test_examples": str(example_path) if made_figure else None}}
    (settings.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nDuration {time.time() - start_time:.1f} s; results in {settings.out}")


if __name__ == "__main__":
    main()
