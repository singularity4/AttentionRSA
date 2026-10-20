"""simulate.py - simulated Euclid VIS stamps of galaxy-scale strong lenses and non-lenses, with arc masks, Einstein
radii and the simulation parameters of every image

AttentionRSA.

The simulated set: 5,000 lenses and 5,000 non-lenses with Einstein radii from 0.3" to 4.5" and neighbouring galaxies,
the second simulated set of EUCLID-SAGE (UNet-Starlight, simulate.py radius-set), with the same forward model,
seeds and images. The U-Nets are trained on it (train_copies.py).

Forward model (lenstronomy)
    Lens galaxy light   elliptical Sersic profile: magnitude 19-22, Sersic index 3-5, Sersic radius 0.5"-1.5".
    Lens mass           singular isothermal ellipsoid (SIE) whose ellipticity is that of the light plus a scatter of
                        0.03 in each component, and an external shear with components drawn from N(0, 0.03^2).
                        The Einstein radius is drawn uniformly in log(theta_E) between 0.3" and 4.5", independently
                        of the light of the galaxy. Non-lenses have an Einstein radius of zero, so that no light is
                        deflected.
    Source light        elliptical Sersic profile: magnitude 22-25, Sersic radius 0.1"-0.4", Sersic index 1-2.5.
                        The source of a lens lies within half an Einstein radius of the lens centre, so that it is
                        imaged into arcs or a ring. Non-lenses contain no source.
    Neighbours          0, 1 or 2 neighbouring galaxies (probabilities 0.4, 0.4 and 0.2) in lenses and non-lenses
                        alike, so that their presence carries no information about the label: offset 1"-5",
                        magnitude 20-25, Sersic radius 0.2"-0.8", Sersic index 0.5-4.
    Observation         the Euclid VIS settings of lenstronomy: Gaussian point-spread function, 0.101" pixels,
                        stamps of 101 x 101 pixels, light profiles evaluated on a grid three times finer than the
                        pixels, zero point 25.72 (electrons per second). Noise: Gaussian background noise of the sky
                        and the detector at the depth of the six-year survey, and Poisson noise of the source and
                        galaxy light. No sky level is added, as in the background-subtracted Q1 mosaics.
    Ellipticity components (e1, e2) are drawn from a normal distribution and limited to +-0.3. All lengths are in
    arcseconds.

Arc mask
    A pixel belongs to the arc mask if the noise-free light of the lensed source, after convolution with the
    point-spread function, exceeds the standard deviation of the background noise in one pixel (0.0109 electrons per
    second). The mask of a non-lens is empty.

Reproducibility
    The parameters of all systems are drawn from one random generator (--seed, default 42) in a fixed order, first
    all lenses and then all non-lenses; the stamps are then shuffled with the same generator. The noise of stamp i is
    drawn with the seed (--noise-seed, i), so that the stamps can be simulated in parallel (--workers) and the result
    does not depend on the number of workers. --parameters-only draws the same parameters without making the images,
    which takes seconds.

Output (.npz, read by train_copies.py)
    images                (N, 101, 101) float32, electrons per second (zero point 25.72)
    mask                  (N, 101, 101) uint8, 1 = arc pixel
    label                 (N,) 1 = lens, 0 = non-lens
    theta_E               (N,) Einstein radius in arcseconds, 0 for non-lenses
    neighbours            (N,) number of neighbouring galaxies
    simulation parameters (N,) each: galaxy_magnitude, galaxy_sersic_radius ("), galaxy_sersic_index,
                          galaxy_ellipticity (sqrt(e1^2 + e2^2) of the light), mass_ellipticity (of the SIE),
                          external_shear (sqrt(gamma1^2 + gamma2^2)), source_magnitude (99 for non-lenses),
                          source_sersic_radius (") and source_offset (" from the centre)

Usage
    python simulate.py --out radius_training_10000.npz
    python simulate.py --n-lens 100 --n-nonlens 100 --out radius_training_200.npz        # small set for tests
    python simulate.py --parameters-only --out radius_training_10000_parameters.npz
"""

import argparse
import multiprocessing
import os
import time

import numpy as np

STAMP_SIZE_PIXELS = 101                 # 0.101" per pixel in the Euclid VIS settings of lenstronomy
SUPERSAMPLING_FACTOR = 3                # light profiles are evaluated on a grid three times finer than the pixels
MAXIMUM_ELLIPTICITY_COMPONENT = 0.3
ABSENT_MAGNITUDE = 99.0                 # a magnitude so faint that the component contributes no light
MAXIMUM_NEIGHBOURS = 2


# ======================================================================================== 1. forward model

def build_simulator(number_of_unlensed_components):
    """The lenstronomy simulator of Euclid VIS stamps: an SIE lens with external shear, one lensed source, and the
    given number of unlensed light components (the lens galaxy and its neighbours). Returns (simulation,
    image_model)."""
    from lenstronomy.SimulationAPI.ObservationConfig.Euclid import Euclid
    from lenstronomy.SimulationAPI.sim_api import SimAPI
    observation = Euclid(band="VIS", psf_type="GAUSSIAN", coadd_years=6).kwargs_single_band()
    model_components = {"lens_model_list": ["SIE", "SHEAR"],
                        "lens_light_model_list": ["SERSIC_ELLIPSE"] * number_of_unlensed_components,
                        "source_light_model_list": ["SERSIC_ELLIPSE"]}
    simulation = SimAPI(num_pix=STAMP_SIZE_PIXELS, kwargs_single_band=observation, kwargs_model=model_components)
    image_model = simulation.image_model_class({"supersampling_factor": SUPERSAMPLING_FACTOR})
    return simulation, image_model


def random_ellipticity(random_generator, spread):
    """Two ellipticity components, each drawn from a normal distribution with standard deviation `spread` and
    limited to +-0.3."""
    first_component = float(np.clip(random_generator.normal(0, spread),
                                    -MAXIMUM_ELLIPTICITY_COMPONENT, MAXIMUM_ELLIPTICITY_COMPONENT))
    second_component = float(np.clip(random_generator.normal(0, spread),
                                     -MAXIMUM_ELLIPTICITY_COMPONENT, MAXIMUM_ELLIPTICITY_COMPONENT))
    return first_component, second_component


def draw_einstein_radius(random_generator, settings):
    """Einstein radius between --theta-e-min and --theta-e-max, uniform in log(theta_E) or in theta_E."""
    if settings.theta_e_distribution == "log-uniform":
        return float(np.exp(random_generator.uniform(np.log(settings.theta_e_min), np.log(settings.theta_e_max))))
    return float(random_generator.uniform(settings.theta_e_min, settings.theta_e_max))


def draw_system(random_generator, is_lens, settings):
    """Parameters of one system, drawn in a fixed order: light and mass of the central galaxy, source, neighbours.
    Light, mass and source are lenstronomy keyword arguments."""
    uniform = random_generator.uniform
    # Central galaxy. The mass follows the light: the ellipticity of the SIE is that of the light plus a scatter.
    light_e1, light_e2 = random_ellipticity(random_generator, spread=0.1)
    lens_light = {"magnitude": uniform(19, 22), "R_sersic": uniform(0.5, 1.5), "n_sersic": uniform(3, 5),
                  "e1": light_e1, "e2": light_e2, "center_x": 0.0, "center_y": 0.0}
    einstein_radius = draw_einstein_radius(random_generator, settings) if is_lens else 0.0
    mass_e1 = light_e1 + random_generator.normal(0, 0.03)
    mass_e2 = light_e2 + random_generator.normal(0, 0.03)
    lens_mass = [{"theta_E": einstein_radius, "e1": mass_e1, "e2": mass_e2, "center_x": 0.0, "center_y": 0.0},
                 {"gamma1": random_generator.normal(0, 0.03), "gamma2": random_generator.normal(0, 0.03)}]

    # Source, within half an Einstein radius of the centre; non-lenses contain no source.
    source_ellipticity = random_ellipticity(random_generator, spread=0.2)
    source_angle = uniform(0, 2 * np.pi)
    source_offset = uniform(0, 0.5) * einstein_radius
    source_magnitude = uniform(22, 25) if is_lens else ABSENT_MAGNITUDE
    source_radius = uniform(0.1, 0.4)
    source = {"magnitude": source_magnitude, "R_sersic": source_radius, "n_sersic": uniform(1, 2.5),
              "e1": source_ellipticity[0], "e2": source_ellipticity[1],
              "center_x": source_offset * np.cos(source_angle), "center_y": source_offset * np.sin(source_angle)}

    # Neighbouring galaxies: position, shape and size are drawn for both possible neighbours; a neighbour that is
    # not present has the magnitude ABSENT_MAGNITUDE.
    number_of_neighbours = int(random_generator.choice(len(settings.neighbour_probabilities),
                                                       p=settings.neighbour_probabilities))
    unlensed_light = [lens_light]
    for neighbour in range(MAXIMUM_NEIGHBOURS):
        neighbour_e1, neighbour_e2 = random_ellipticity(random_generator, spread=0.2)
        offset, angle = uniform(1.0, 5.0), uniform(0, 2 * np.pi)
        magnitude = uniform(20, 25) if neighbour < number_of_neighbours else ABSENT_MAGNITUDE
        unlensed_light.append({"magnitude": magnitude, "R_sersic": uniform(0.2, 0.8), "n_sersic": uniform(0.5, 4),
                               "e1": neighbour_e1, "e2": neighbour_e2,
                               "center_x": offset * np.cos(angle), "center_y": offset * np.sin(angle)})
    return {"lens_light": unlensed_light, "lens_mass": lens_mass, "source_light": [source],
            "theta_E": einstein_radius, "label": int(is_lens), "neighbours": number_of_neighbours}


def simulation_parameters(systems):
    """The simulation parameters of the central galaxy, its mass and the source of every system (read from the
    systems, so that the random draws and the images do not change). Non-lenses have no source (magnitude 99)."""
    galaxy = [system["lens_light"][0] for system in systems]
    mass = [system["lens_mass"] for system in systems]
    source = [system["source_light"][0] for system in systems]
    def values(components, read):
        return np.array([read(component) for component in components], np.float32)

    return {"galaxy_magnitude": values(galaxy, lambda light: light["magnitude"]),
            "galaxy_sersic_radius": values(galaxy, lambda light: light["R_sersic"]),
            "galaxy_sersic_index": values(galaxy, lambda light: light["n_sersic"]),
            "galaxy_ellipticity": values(galaxy, lambda light: np.hypot(light["e1"], light["e2"])),
            "mass_ellipticity": values(mass, lambda sie_and_shear: np.hypot(sie_and_shear[0]["e1"],
                                                                            sie_and_shear[0]["e2"])),
            "external_shear": values(mass, lambda sie_and_shear: np.hypot(sie_and_shear[1]["gamma1"],
                                                                          sie_and_shear[1]["gamma2"])),
            "source_magnitude": values(source, lambda light: light["magnitude"]),
            "source_sersic_radius": values(source, lambda light: light["R_sersic"]),
            "source_offset": values(source, lambda light: np.hypot(light["center_x"], light["center_y"]))}


# ============================================================================================ 2. the images

worker_simulator = None                 # one simulator in each worker process


def start_worker():
    """Builds the lenstronomy simulator once in each worker process; the worker simulates all its stamps with it."""
    global worker_simulator
    worker_simulator = build_simulator(number_of_unlensed_components=1 + MAXIMUM_NEIGHBOURS)


def simulate_stamp(task):
    """Observed stamp and arc mask of one system. task = (index, system, noise_seed)."""
    index, system, noise_seed = task
    simulation, image_model = worker_simulator
    lens_light, source, _ = simulation.magnitude2amplitude(system["lens_light"], system["source_light"])
    source_image = image_model.image(system["lens_mass"], source, lens_light, lens_light_add=False)
    galaxy_image = image_model.image(system["lens_mass"], source, lens_light, source_add=False)
    noise_free_image = source_image + galaxy_image
    np.random.seed([noise_seed, index])          # lenstronomy draws the noise from NumPy's global random generator
    image = noise_free_image + simulation.noise_for_model(model=noise_free_image)
    arc_mask = (source_image > float(simulation.background_noise)) & bool(system["label"])
    return index, image.astype(np.float32), arc_mask.astype(np.uint8)


def simulate(settings):
    """Stamps, arc masks, labels, Einstein radii, numbers of neighbours and simulation parameters of the simulated set
    (with --parameters-only, all but the stamps and arc masks)."""
    random_generator = np.random.default_rng(settings.seed)
    systems = [draw_system(random_generator, is_lens, settings)
               for is_lens, count in ((True, settings.n_lens), (False, settings.n_nonlens)) for _ in range(count)]
    order = random_generator.permutation(len(systems))
    arrays = {"label": np.array([system["label"] for system in systems])[order],
              "theta_E": np.array([system["theta_E"] for system in systems], np.float32)[order],
              "neighbours": np.array([system["neighbours"] for system in systems], np.int8)[order],
              **{name: values[order] for name, values in simulation_parameters(systems).items()}}
    if settings.parameters_only:
        return arrays

    tasks = [(index, system, settings.noise_seed) for index, system in enumerate(systems)]
    images = np.empty((len(systems), STAMP_SIZE_PIXELS, STAMP_SIZE_PIXELS), np.float32)
    masks = np.empty(images.shape, np.uint8)
    workers = max(1, settings.workers)
    start = time.time()
    if workers == 1:
        start_worker()
        results = map(simulate_stamp, tasks)
    else:
        pool = multiprocessing.get_context("fork").Pool(workers, initializer=start_worker)
        results = pool.imap_unordered(simulate_stamp, tasks, chunksize=16)
    for finished, (index, image, mask) in enumerate(results, start=1):
        images[index], masks[index] = image, mask
        if finished % 2000 == 0 or finished == len(tasks):
            elapsed = time.time() - start
            print(f"  {finished:,} of {len(tasks):,} stamps, {elapsed / 60:.1f} min "
                  f"(about {elapsed / finished * (len(tasks) - finished) / 60:.0f} min left)", flush=True)
    if workers > 1:
        pool.close()
        pool.join()
    return {"images": images[order], "mask": masks[order], **arrays}


# ========================================================================================== 3. command line

def parse_arguments():
    argument_parser = argparse.ArgumentParser(description="Simulated Euclid VIS stamps of lenses and non-lenses")
    argument_parser.add_argument("--n-lens", type=int, default=5000, help="number of lenses (default 5000)")
    argument_parser.add_argument("--n-nonlens", type=int, default=5000, help="number of non-lenses (default 5000)")
    argument_parser.add_argument("--theta-e-min", type=float, default=0.3, help="smallest Einstein radius (0.3\")")
    argument_parser.add_argument("--theta-e-max", type=float, default=4.5, help="largest Einstein radius (4.5\")")
    argument_parser.add_argument("--theta-e-distribution", choices=["log-uniform", "uniform"], default="log-uniform")
    argument_parser.add_argument("--neighbour-probabilities", type=float, nargs=MAXIMUM_NEIGHBOURS + 1,
                                 default=[0.4, 0.4, 0.2], help="probabilities of 0, 1 and 2 neighbours")
    argument_parser.add_argument("--seed", type=int, default=42, help="seed of the parameters (default 42)")
    argument_parser.add_argument("--noise-seed", type=int, default=1, help="seed of the noise (default 1)")
    argument_parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="parallel processes")
    argument_parser.add_argument("--parameters-only", action="store_true",
                                 help="only the labels, Einstein radii and simulation parameters, without the images")
    argument_parser.add_argument("--out", default="radius_training_10000.npz", help="output file")
    settings = argument_parser.parse_args()
    settings.neighbour_probabilities = np.asarray(settings.neighbour_probabilities, dtype=float)
    if abs(settings.neighbour_probabilities.sum() - 1) > 1e-6:
        raise SystemExit("--neighbour-probabilities must add up to 1")
    return settings


def main():
    settings = parse_arguments()
    start = time.time()
    arrays = simulate(settings)
    np.savez_compressed(settings.out, **arrays)
    lens_radii = arrays["theta_E"][arrays["label"] == 1]
    print(f"Saved {settings.out}: {len(arrays['label']):,} systems ({int(arrays['label'].sum()):,} lenses), "
          f"Einstein radii {lens_radii.min():.2f}\"-{lens_radii.max():.2f}\" (median {np.median(lens_radii):.2f}\")"
          f"{', parameters only' if settings.parameters_only else ''}; {(time.time() - start) / 60:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
