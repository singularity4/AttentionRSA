"""data.py - reading and preparing the simulated images and the real EUCLID-SAGE stamps for the U-Nets

AttentionRSA.

Folders
    Working folder: EUCLID_SAGE_DATA_DIR if set, otherwise the folder from which a script is run. All results are
    written to results/ in the working folder. The tables and stamps of the 1k benchmarks are read from
    EUCLID_SAGE_BENCHMARK_DIR, the folder benchmarks/ of the EUCLID-SAGE repository
    (https://github.com/singularity4/EUCLID-SAGE), or, if it is not set, from benchmarks/ in the working folder.

Simulated images (made by simulate.py)
    Pixel values in electrons per second (zero point 25.72, the Euclid VIS setting of lenstronomy). The images
    contain background noise but no sky level, like the background-subtracted Q1 mosaics.
    Split: stratified into 80% training, 10% validation and 10% test images (8,000 / 1,000 / 1,000 of the 10,000
    images). 20% of the images are held out with the split seed and divided equally into validation and test images
    with the split seed + 1. The split depends only on the labels and the seed, so it is the same for every network
    trained on the same images.

Real stamps (EUCLID-SAGE)
    101 x 101 pixels of 0.1", cut from the background-subtracted Q1 MER VIS mosaics, in ADU per second with the zero
    point MAGZERO of the FITS header (24.6). The stamps are used as delivered: no further background subtraction,
    masking, clipping, resampling or rotation.
    Unit conversion: the pixel values of a source of magnitude m are proportional to 10^(-0.4 (m - ZP)), so the same
    source has values larger by 10^(0.4 (25.72 - MAGZERO)) = 2.81 in the units of the simulations. Every stamp is
    multiplied by this factor, computed from its own MAGZERO; this changes the unit, not the calibration.
    Splits: five-fold cross-validation, so that every stamp is tested once. Inside each fold the training stamps are
    split again into training and validation stamps (80 / 20); the validation stamps are used to choose the epoch and
    the learning rate.

Input transform (the same for both kinds of images)
    z = (asinh(x / s) - mean) / std. The asinh is approximately linear for faint pixels (arcs and noise) and
    logarithmic for bright galaxy centres, so that faint arcs and bright galaxies share one scale. s = 0.02 electrons
    per second, about twice the standard deviation of the background noise of the simulations. mean and std are
    computed from the transformed training images only, never from validation or test images: from the simulated
    training images when training on simulations (they are stored with the network), and from the training stamps of
    each fold on the real stamps. For the real stamps, only s is taken from the network trained on simulations.

Augmentation
    Training images only: a rotation by a random multiple of 90 degrees, followed by a reflection with probability
    1/2 (the eight symmetries of the square). Lens configurations have no preferred orientation or handedness on the
    sky; the arc mask of a simulated image is transformed together with the image.
"""

import io
import os
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from astropy.io import fits
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split
from torch.utils.data import Dataset

# ================================================================================================== 1. folders

WORKING_FOLDER = Path(os.environ.get("EUCLID_SAGE_DATA_DIR", ".")).resolve()
RESULTS_FOLDER = WORKING_FOLDER / "results"
BENCHMARK_FOLDER = Path(os.environ.get("EUCLID_SAGE_BENCHMARK_DIR", WORKING_FOLDER / "benchmarks")).resolve()
SIMULATION_ZERO_POINT = 25.72           # zero point of the Euclid VIS settings of lenstronomy (electrons per second)


# ============================================================================================= 2. random seeds

def set_random_seeds(seed):
    """Seed the random generators of Python, NumPy and PyTorch, which together determine the initial weights, the
    order of the training images, the augmentation and the dropout."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_data_loader_worker(worker_id):
    """Seed the random generators of a process that loads training images in parallel, so that the augmentation
    remains reproducible (PyTorch gives each such process its own seed)."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ========================================================================================= 3. simulated images

def read_simulated_set(path):
    """All arrays of a simulated set: a .npz file, or a .zip file that contains exactly one .npz file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            return {name: arrays[name] for name in arrays.files}
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as archive:
            npz_names = [name for name in archive.namelist() if name.lower().endswith(".npz")]
            if len(npz_names) != 1:
                raise ValueError(f"Expected exactly one .npz file in {path}, found {len(npz_names)}: {npz_names}")
            npz_bytes = archive.read(npz_names[0])
        with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as arrays:
            return {name: arrays[name] for name in arrays.files}
    raise ValueError("The simulated set must be a .npz or .zip file")


def check_simulated_set(arrays):
    """Check the shapes, the labels and the arc masks: every lens has a non-empty arc mask and every non-lens an
    empty one. Returns (images, arc_masks, labels)."""
    missing = {"images", "mask", "label"} - set(arrays)
    if missing:
        raise ValueError(f"Missing arrays: {sorted(missing)}")
    images = np.asarray(arrays["images"], dtype=np.float32)
    arc_masks = np.asarray(arrays["mask"], dtype=np.uint8)
    labels = np.asarray(arrays["label"], dtype=np.int64)
    if images.ndim != 3 or images.shape != arc_masks.shape:
        raise ValueError(f"Expected images and masks of the same shape (N, H, W), got {images.shape} and "
                         f"{arc_masks.shape}")
    if len(labels) != len(images):
        raise ValueError("The number of labels differs from the number of images")
    if not set(np.unique(labels)).issubset({0, 1}) or not set(np.unique(arc_masks)).issubset({0, 1}):
        raise ValueError("Expected labels and mask values 0 and 1")
    if np.any(arc_masks[labels == 0] != 0):
        raise ValueError("At least one non-lens has a non-empty arc mask")
    if np.any(arc_masks[labels == 1].reshape(np.sum(labels == 1), -1).sum(1) == 0):
        raise ValueError("At least one lens has an empty arc mask")
    return images, arc_masks, labels


def split_simulated_set(labels, seed):
    """Stratified split into 80% training, 10% validation and 10% test images.
    Returns (training, validation, test) indices."""
    all_indices = np.arange(len(labels))
    training_indices, held_out_indices = train_test_split(all_indices, test_size=0.20, stratify=labels,
                                                          random_state=seed)
    validation_indices, test_indices = train_test_split(held_out_indices, test_size=0.50,
                                                        stratify=labels[held_out_indices], random_state=seed + 1)
    return training_indices, validation_indices, test_indices


class IntensityTransform:
    """z = (asinh(x / scale) - mean) / std, with mean and std of the transformed training images. The three values
    are stored with the network (as_dict) and reused when the network is applied."""

    def __init__(self, scale, mean, std):
        self.scale = float(scale)
        self.mean = float(mean)
        self.std = float(max(std, 1e-6))

    @classmethod
    def fit(cls, training_images, scale):
        """The transform with mean and std of the asinh-transformed training images."""
        transformed = np.arcsinh(training_images.astype(np.float32) / float(scale))
        return cls(scale=scale, mean=float(transformed.mean()), std=float(transformed.std()))

    def __call__(self, pixel_values):
        transformed = np.arcsinh(pixel_values.astype(np.float32) / self.scale)
        return ((transformed - self.mean) / self.std).astype(np.float32)

    def as_dict(self):
        return {"asinh_scale": self.scale, "mean": self.mean, "std": self.std}


class ArcSegmentationDataset(Dataset):
    """Simulated images after the input transform, with their arc masks, labels and indices. With augment=True,
    image and arc mask receive the same random rotation and reflection."""

    def __init__(self, images, arc_masks, labels, indices, transform, augment):
        self.images = images
        self.arc_masks = arc_masks
        self.labels = labels
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform
        self.augment = bool(augment)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        image_index = int(self.indices[position])
        image = self.transform(self.images[image_index])
        arc_mask = self.arc_masks[image_index].astype(np.float32)
        if self.augment:
            quarter_turns = random.randrange(4)            # rotation by 0, 90, 180 or 270 degrees
            if quarter_turns:
                image = np.rot90(image, quarter_turns)
                arc_mask = np.rot90(arc_mask, quarter_turns)
            if random.random() < 0.5:                      # reflection with probability 1/2
                image = np.fliplr(image)
                arc_mask = np.fliplr(arc_mask)
        # rot90 and fliplr return views with negative strides, which PyTorch does not accept; contiguous copies are made
        image = torch.from_numpy(np.ascontiguousarray(image[None, ...])).float()
        arc_mask = torch.from_numpy(np.ascontiguousarray(arc_mask[None, ...])).float()
        return image, arc_mask, torch.tensor(int(self.labels[image_index])), torch.tensor(image_index)


# ============================================================================================= 4. real stamps

def read_benchmark_table(benchmark_name, folder=BENCHMARK_FOLDER):
    """The table of a benchmark: one row per stamp, with stamp_file, label, grade, object_id and catalogue values."""
    return pd.read_csv(Path(folder) / f"{benchmark_name}.csv")


def load_stamps_in_simulation_units(benchmark_table, folder=BENCHMARK_FOLDER):
    """All stamps of a benchmark table as one array, multiplied by 10^(0.4 (25.72 - MAGZERO)) to convert them to the
    units of the simulations. The paths in stamp_file are relative to folder."""
    stamps = []
    for stamp_file in benchmark_table.stamp_file:
        with fits.open(Path(folder) / stamp_file) as fits_file:
            pixel_values = fits_file[0].data.astype(np.float32)
            stamp_zero_point = float(fits_file[0].header["MAGZERO"])
        stamps.append(pixel_values * 10 ** (0.4 * (SIMULATION_ZERO_POINT - stamp_zero_point)))
    return np.stack(stamps)


def asinh_transform(stamps, asinh_scale):
    """asinh(x / s), the first part of the input transform; the standardisation with the mean and standard deviation
    of the training stamps of a fold follows in the data sets."""
    return np.arcsinh(stamps / asinh_scale).astype(np.float32)


class StampDataset(Dataset):
    """asinh-transformed stamps, standardised with the mean and standard deviation of the training stamps of the
    fold, with their targets (a label or an Einstein radius) and positions. With apply_symmetries=True, every stamp
    receives a random rotation and reflection."""

    def __init__(self, transformed_stamps, targets, mean, standard_deviation, apply_symmetries):
        self.transformed_stamps = transformed_stamps
        self.targets = targets
        self.mean = float(mean)
        self.standard_deviation = max(float(standard_deviation), 1e-8)
        self.apply_symmetries = apply_symmetries

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        image = self.transformed_stamps[index]
        if self.apply_symmetries:
            image = np.rot90(image, np.random.randint(4))  # rotation by 0, 90, 180 or 270 degrees
            if np.random.rand() < 0.5:                     # reflection with probability 1/2
                image = np.fliplr(image)
        image = (np.ascontiguousarray(image) - self.mean) / self.standard_deviation
        return torch.from_numpy(image[None].astype(np.float32)), self.targets[index], index


def cross_validation_folds(labels, folds, seed):
    """Stratified k-fold cross-validation: yields (fold number from 1, training rows, test rows)."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for fold, (training_rows, test_rows) in enumerate(splitter.split(np.zeros(len(labels)), labels), start=1):
        yield fold, training_rows, test_rows


def inner_validation_split(training_rows, labels, fraction, seed):
    """Split the training rows of one fold into training and validation rows, with the lens fraction kept.
    Returns (training rows, validation rows)."""
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    inner_training, inner_validation = next(splitter.split(np.zeros(len(training_rows)), labels[training_rows]))
    return training_rows[inner_training], training_rows[inner_validation]
