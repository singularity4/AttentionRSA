"""metrics.py - the metrics of the U-Nets on simulated images and of the fine-tuned networks on real stamps

AttentionRSA.

Arc segmentation (simulated images)
    A pixel belongs to the predicted arc mask if its arc probability is at least the threshold (0.5). For every lens
    the overlap of the predicted mask P and the true mask T is measured by the Dice coefficient,
    2 |P ∩ T| / (|P| + |T|), and by the intersection over union, |P ∩ T| / |P ∪ T|; both are summarised over the
    lenses by their mean and median. A constant of 1e-8 added to numerator and denominator makes both measures equal
    to 1 when the two masks are empty. For non-lenses, whose true mask is empty, the number of predicted arc pixels is
    reported.
    Lens decision from the segmentation: a stamp is classified as a lens if at least 40 pixels have an arc probability
    of at least 0.5; the decisions are summarised by the balanced accuracy and the confusion matrix.

Einstein radius
    The error of the estimated Einstein radius over the lenses, in arcseconds: mean and median absolute error, and
    mean signed error.

Lens classification (real stamps)
    AUROC of the lens probability: the probability that a randomly chosen lens receives a higher score than a randomly
    chosen non-lens. For the decisions of the classifier (a lens if the lens probability is above 0.5, that is, if
    the lens output is the larger of the two):
    completeness, the fraction of lenses classified as lenses, also separately for grade A and grade B lenses;
    false-alarm rate, the fraction of non-lenses classified as lenses; precision on the benchmark; balanced accuracy
    and the confusion matrix.
"""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score


# ======================================================================================== 1. arc segmentation

def segmentation_metrics(probabilities, true_masks, labels, threshold, min_pixels=40):
    """Overlap of predicted and true arc masks for the lenses, predicted arc pixels for the non-lenses, and the lens
    decisions of the rule 'at least min_pixels pixels with arc probability >= threshold'."""
    predicted_masks = probabilities >= threshold
    true_masks = true_masks.astype(bool)
    labels = labels.astype(int)
    is_lens = labels == 1
    is_nonlens = ~is_lens

    intersection = np.logical_and(predicted_masks, true_masks).sum(axis=(1, 2))
    union = np.logical_or(predicted_masks, true_masks).sum(axis=(1, 2))
    predicted_pixels = predicted_masks.sum(axis=(1, 2))
    true_pixels = true_masks.sum(axis=(1, 2))
    dice = (2 * intersection + 1e-8) / (predicted_pixels + true_pixels + 1e-8)
    intersection_over_union = (intersection + 1e-8) / (union + 1e-8)
    predicted_lens = (predicted_pixels >= int(min_pixels)).astype(int)

    def summary(function, values, selection):
        return float(function(values[selection])) if np.any(selection) else float("nan")

    return {"threshold": float(threshold),
            "min_pixels_for_lens": int(min_pixels),
            "lens_dice_mean": summary(np.mean, dice, is_lens),
            "lens_dice_median": summary(np.median, dice, is_lens),
            "lens_iou_mean": summary(np.mean, intersection_over_union, is_lens),
            "lens_iou_median": summary(np.median, intersection_over_union, is_lens),
            "nonlens_mean_predicted_pixels": summary(np.mean, predicted_pixels, is_nonlens),
            "nonlens_zero_mask_fraction": summary(np.mean, predicted_pixels == 0, is_nonlens),
            "mask_based_lens_balanced_accuracy": float(balanced_accuracy_score(labels, predicted_lens)),
            "mask_based_lens_confusion_0_1": confusion_matrix(labels, predicted_lens, labels=[0, 1]).tolist()}


def mean_lens_dice(probabilities, true_masks, labels, threshold=0.5):
    """Mean Dice coefficient of the predicted arc masks over the lenses: the criterion by which the epoch is chosen
    in the training on simulations."""
    predicted_masks = probabilities >= threshold
    true_masks = true_masks.astype(bool)
    is_lens = labels == 1
    if not np.any(is_lens):
        return float("nan")
    intersection = np.logical_and(predicted_masks[is_lens], true_masks[is_lens]).sum(axis=(1, 2))
    predicted_pixels = predicted_masks[is_lens].sum(axis=(1, 2))
    true_pixels = true_masks[is_lens].sum(axis=(1, 2))
    dice = (2 * intersection + 1e-8) / (predicted_pixels + true_pixels + 1e-8)
    return float(dice.mean())


# ========================================================================================= 2. Einstein radius

def einstein_radius_metrics(true_radii, estimated_radii, labels):
    """Errors of the estimated Einstein radius over the lenses."""
    is_lens = np.asarray(labels) == 1
    error = np.asarray(estimated_radii, dtype=float)[is_lens] - np.asarray(true_radii, dtype=float)[is_lens]
    return {"lenses": int(is_lens.sum()),
            "mean_absolute_error": float(np.mean(np.abs(error))),
            "median_absolute_error": float(np.median(np.abs(error))),
            "mean_error": float(np.mean(error))}


# ===================================================================================== 3. lens classification

def area_under_roc(labels, probabilities):
    """AUROC of the lens probabilities; NaN if only one class is present."""
    return float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else float("nan")


def classification_metrics(predictions):
    """AUROC, completeness, false alarms, precision, balanced accuracy and confusion matrix, from a table with the
    columns label, grade, lens_probability and predicted_lens."""
    labels, decisions = predictions.label.to_numpy(), predictions.predicted_lens.to_numpy()
    lenses, nonlenses = predictions[predictions.label == 1], predictions[predictions.label == 0]
    found = int(lenses.predicted_lens.sum())
    false_alarms = int(nonlenses.predicted_lens.sum())
    metrics = {"auroc": area_under_roc(labels, predictions.lens_probability.to_numpy()),
               "lenses_found": found, "lenses": len(lenses), "completeness": round(found / len(lenses), 4),
               "false_alarms": false_alarms, "nonlenses": len(nonlenses),
               "false_alarm_rate": round(false_alarms / len(nonlenses), 5),
               "precision_in_benchmark": round(found / (found + false_alarms), 4) if found + false_alarms else None,
               "balanced_accuracy": float(balanced_accuracy_score(labels, decisions)),
               "confusion_matrix": confusion_matrix(labels, decisions, labels=[0, 1]).tolist()}
    for grade in ("A", "B"):
        graded = lenses[lenses.grade == grade]
        metrics[f"completeness_grade_{grade}"] = round(float(graded.predicted_lens.mean()), 4) if len(graded) else None
    return metrics
