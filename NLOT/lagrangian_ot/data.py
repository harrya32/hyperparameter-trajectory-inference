import os
import jax
import jax.numpy as jnp
import numpy as np
import torch as th

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

def _infer_2d_bounds(samples, padding=0.05):
    """Infer plotting bounds from the first two ambient dimensions."""
    samples_np = np.asarray(samples)
    if samples_np.shape[-1] < 2:
        raise ValueError(
            "Automatic bound inference requires at least 2 ambient dimensions. "
            "Provide explicit `plot_bounds=[xmin, xmax, ymin, ymax]`."
        )

    x_vals = samples_np[..., 0]
    y_vals = samples_np[..., 1]

    x_min, x_max = float(np.min(x_vals)), float(np.max(x_vals))
    y_min, y_max = float(np.min(y_vals)), float(np.max(y_vals))

    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    x_pad = padding * x_span
    y_pad = padding * y_span

    xbounds = (x_min - x_pad, x_max + x_pad)
    ybounds = (y_min - y_pad, y_max + y_pad)
    bounds = (
        jnp.array((xbounds[0], ybounds[0])),
        jnp.array((xbounds[1], ybounds[1])),
    )
    return bounds, xbounds, ybounds


def _bounds_from_custom(custom_bounds):
    if len(custom_bounds) != 4:
        raise ValueError(
            "Expected `plot_bounds` to have 4 values: [xmin, xmax, ymin, ymax]."
        )
    x_min, x_max, y_min, y_max = [float(v) for v in custom_bounds]
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Invalid `plot_bounds`: require xmin < xmax and ymin < ymax.")

    xbounds = (x_min, x_max)
    ybounds = (y_min, y_max)
    bounds = (
        jnp.array((xbounds[0], ybounds[0])),
        jnp.array((xbounds[1], ybounds[1])),
    )
    return bounds, xbounds, ybounds


def get_bounds(name, samples=None, custom_bounds=None):
    if custom_bounds is not None:
        return _bounds_from_custom(custom_bounds)

    if name == "conditional_semicircles":
        xbounds = (-2.5, 2.5)
        ybounds = (-1.5, 1.5)
        bounds = (
            jnp.array((xbounds[0], ybounds[0])),
            jnp.array((xbounds[1], ybounds[1])),
        )
    elif name == "reward_weighting_data":
        xbounds = (0, 10)
        ybounds = (0, 8)
        bounds = (
            jnp.array((xbounds[0], ybounds[0])),
            jnp.array((xbounds[1], ybounds[1])),
        )
    elif name == "reacher_data":
        bounds = (-1, 1)
        xbounds = ybounds = bounds
    elif name == "quantile_data":
        bounds = (25, 55)
        xbounds = ybounds = bounds
    elif name == "reward_weighting_hinge_data":
        xbounds = (0, 10)
        ybounds = (0, 8)
        bounds = (
            jnp.array((xbounds[0], ybounds[0])),
            jnp.array((xbounds[1], ybounds[1])),
        )
    elif name == "2moons_dropout":
        bounds = (-4, 4)
        xbounds = ybounds = bounds
    else:
        if samples is not None:
            return _infer_2d_bounds(samples=samples)
        raise ValueError(
            f"Invalid data choice: {name}. "
            "For custom datasets, pass `dataset_path=...` and either "
            "`plot_bounds=[xmin, xmax, ymin, ymax]` or keep defaults to auto-infer bounds."
        )

    return bounds, xbounds, ybounds

def get_samplers(dataset_name, num_pairs_requested=None, dataset_path=None):
    """Load datasets, optionally selecting a subset of pairs.

    Args:
        dataset_name: Name of the dataset (e.g., 'conditional_semicircles').
        num_pairs_requested: The desired number of evenly spaced pairs. If None
                             or >= total available pairs, all pairs are used.
        dataset_path: Optional path to a custom `.pt` dataset file. When provided,
            no built-in dataset-specific preprocessing is applied.

    Returns:
        A list of sampler iterators corresponding to the selected timepoints.
    """
    paths = {
        "conditional_semicircles": "conditional_semicircles.pt",
        "reward_weighting_data": "reward_weighting_data_0_10.pt",
        "reacher_data": "reacher_data.pt",
        "quantile_data" : "quantile_data_new.pt",
        "reward_weighting_hinge_data": "reward_weighting_hinge_data.pt",
        "2moons_dropout": "diffusion_2moons_dropout.pt"
    }

    if dataset_path is not None:
        fname = os.path.abspath(dataset_path)
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Could not find custom dataset file: {fname}")
    else:
        if dataset_name not in paths:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Use one of {list(paths.keys())} "
                "or pass `dataset_path=/path/to/data.pt`."
            )
        fname = os.path.join(SCRIPT_PATH, "..", "data", paths[dataset_name])

    dataset = th.load(fname, map_location="cpu", weights_only=False)
    if isinstance(dataset, th.Tensor):
        dataset = dataset.detach()

    # Keep legacy preprocessing for built-in datasets only.
    if dataset_path is None and dataset_name == "reward_weighting_data":
        dataset = dataset[[0, 5, 10], :1000, :]
        #dataset = dataset[[0,10], :1000, :]
        dataset = jnp.asarray(dataset)
        #add tiny amount of noise for spline stability
        noise = 0.001 * jax.random.normal(jax.random.PRNGKey(0), dataset[:, :, :2].shape)
        dataset = dataset.at[:, :, :2].set(dataset[:, :, :2] + noise)
        dataset = dataset.at[:, :, 0].set(jnp.clip(dataset[:, :, 0], 0, 10))
        dataset = dataset.at[:, :, 1].set(jnp.clip(dataset[:, :, 1], 0, 8))
    elif dataset_path is None and dataset_name == "reward_weighting_hinge_data":
        dataset = dataset[[0, 1, 2], :1000, :]
        dataset = jnp.asarray(dataset)
        noise = 0.001 * jax.random.normal(jax.random.PRNGKey(0), dataset[:, :, :2].shape)
        dataset = dataset.at[:, :, :2].set(dataset[:, :, :2] + noise)
        dataset = dataset.at[:, :, 0].set(jnp.clip(dataset[:, :, 0], 0, 10))
        dataset = dataset.at[:, :, 1].set(jnp.clip(dataset[:, :, 1], 0, 8))
    elif dataset_path is None and dataset_name == "quantile_data":
        dataset = jnp.asarray(dataset)
        dataset = dataset[jnp.array([0, -1])]
        dataset_ambient = dataset[:, :1200, 12:]
        dataset_conditioning = dataset[:, :1200, :12]
        dataset = jnp.concatenate((dataset_ambient, dataset_conditioning), axis=2)
    elif dataset_path is None and dataset_name == "2moons_dropout":
         dataset = dataset[[0,5,-1], :, :]
         dataset = jnp.asarray(dataset)


    print('Dataset shape:', dataset.shape)
    dataset = jnp.asarray(dataset)
    if dataset.ndim != 3:
        raise ValueError(
            f"Expected dataset with shape [num_timepoints, num_samples, D+C], got {dataset.shape}."
        )


    total_timepoints = dataset.shape[0]
    total_pairs = total_timepoints - 1

    if num_pairs_requested is not None and num_pairs_requested > 0 and num_pairs_requested < total_pairs:
        print(f"Selecting {num_pairs_requested} pairs evenly spaced from {total_pairs} total pairs.")
        num_timepoints_to_select = num_pairs_requested + 1
        selected_indices_float = np.linspace(0, total_timepoints - 1, num=num_timepoints_to_select)
        selected_timepoint_indices = np.round(selected_indices_float).astype(int)
        selected_timepoint_indices = np.unique(selected_timepoint_indices)
        
        if len(selected_timepoint_indices) < num_timepoints_to_select:
             selected_timepoint_indices = np.linspace(0, total_timepoints - 1, num=num_timepoints_to_select).astype(int)
             selected_timepoint_indices = np.unique(selected_timepoint_indices)
             print(f"Adjusted selected timepoint indices: {selected_timepoint_indices}")


        print(f"Using timepoint indices: {selected_timepoint_indices}")
        dataset = dataset[selected_timepoint_indices]
    else:
        print(f"Using all {total_pairs} available pairs.")
        selected_timepoint_indices = np.arange(total_timepoints)


    samplers = [
        iter(sampler_from_data(dataset[t])) for t in range(dataset.shape[0])
    ]

    return samplers

def sampler_from_data(data, batch_size=None):
    while True:
        if batch_size is None:
            yield data
        else:
            idx = np.random.choice(data.shape[0], batch_size, replace=False)
            yield data[idx]
