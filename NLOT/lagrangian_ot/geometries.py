import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn
import functools
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Dict
from lagrangian_ot import (
    metrics,
    geodesics,
    splines,
    spline_amortizer,
)

class DistanceModes:
    GEODESIC = "geodesic"
    SQUARED_GEODESIC = "sq_geodesic"
    LAGRANGIAN = "lagrangian"


def get(
    name, 
    geometry_kwargs, 
    D=2, 
    C=0, 
    categorical=False, 
    num_categories=0,
    lagrangian_potential_initializer_fn=None):
    
    if name == "sq_euclidean_manifold":
        return MetricManifold(
            distance_mode=DistanceModes.SQUARED_GEODESIC,
            metric_initializer_fn=metrics.EuclideanMetric,
            C=C,
            D=D,
            bounds=(-2,2),
            categorical=categorical,
            num_categories=num_categories,
            lagrangian_potential_initializer_fn=lagrangian_potential_initializer_fn,
            **geometry_kwargs,
        )
    elif name == "neural_net_metric":
        return MetricManifold(
            bounds=(-2, 2),
            distance_mode=DistanceModes.SQUARED_GEODESIC,
            metric_initializer_fn=metrics.NeuralNetMetric,
            D=D,
            C=C,
            categorical=categorical,
            num_categories=num_categories,
            lagrangian_potential_initializer_fn=lagrangian_potential_initializer_fn,
            **geometry_kwargs,
        )
    elif name == "neural_net_metric_eig":
        return MetricManifold(
            bounds=(-2, 2),
            distance_mode=DistanceModes.SQUARED_GEODESIC,
            metric_initializer_fn=metrics.NeuralNetMetricEig,
            D=D,
            C=C,
            categorical=categorical,
            num_categories=num_categories,
            lagrangian_potential_initializer_fn=lagrangian_potential_initializer_fn,
            **geometry_kwargs,
        )
    else:
        raise ValueError(f"Unknown geometry: {name}")


@dataclass
class GeometryBase(ABC, nn.Module):
    D: int = 2  # dimension of the ambient space
    C: int = 0  # conditional dimension
    bounds: Tuple = (-2, 2)  # bounds of the measures

    # for 2d geometries
    xbounds: Tuple = None
    ybounds: Tuple = None

    @abstractmethod
    def cost(self, x, y):
        pass

    @abstractmethod
    def path(self, x, y, num_points=20):
        pass

    @abstractmethod
    def project(self, x):
        pass

    def add_plot_background(self, params, ax, xlims, ylims=None, **kwargs):
        pass

@dataclass
class MetricManifold(GeometryBase):
    distance_mode: DistanceModes = DistanceModes.SQUARED_GEODESIC
    metric_initializer_fn: Callable = metrics.EuclideanMetric
    spline_model_initializer_fn: Callable = spline_amortizer.SplineMLP
    lagrangian_potential_initializer_fn: Optional[Callable] = None
    spline_solver_kwargs: Optional[Dict] = None
    samples: Optional[jnp.ndarray] = None
    land_kwargs: Optional[Dict] = None
    rbf_kwargs: Optional[Dict] = None
    categorical: Optional[bool] = False
    num_categories: Optional[int] = 0

    def __post_init__(self):

        if self.spline_solver_kwargs is None:
            self.spline_solver_kwargs = {}

        self.spline_geodesic_solver = geodesics.SplineSolver(
            D=self.D, 
            **self.spline_solver_kwargs
        )

        self.spline_amortizer = spline_amortizer.SplineAmortizer(
            self, 
            self.spline_geodesic_solver, 
            D=self.D, 
            C=self.C
        )

        super().__post_init__()

    def setup(self):

        if self.samples is not None and self.land_kwargs is not None:
            self.metric_module = self.metric_initializer_fn(
                D=self.D,
                C=self.C, 
                samples=self.samples, 
                categorical=self.categorical, 
                num_categories=self.num_categories, 
                **self.land_kwargs
                )
        elif self.samples is not None and self.rbf_kwargs is not None:
            self.metric_module = self.metric_initializer_fn(
                D=self.D, 
                C=self.C, 
                samples=self.samples,
                categorical=self.categorical,
                num_categories=self.num_categories,
                **self.rbf_kwargs
                )
        else:
            self.metric_module = self.metric_initializer_fn(
                D=self.D, 
                C=self.C,
                categorical=self.categorical,
                num_categories=self.num_categories
                )
        
        if self.lagrangian_potential_initializer_fn is not None:
            self.lagrangian_potential_module = self.lagrangian_potential_initializer_fn

        self.spline_model = self.spline_model_initializer_fn(
            out_dims=self.spline_geodesic_solver.num_spline_params, 
            D=self.D, 
            C=self.C, 
            categorical=self.categorical, 
            num_categories=self.num_categories
        )

    def predict_spline_params(self, x, y):
        return self.spline_model(x, y)

    def metric(self, x):
        return self.metric_module(x)

    def lagrangian_potential(self, x):
        return self.lagrangian_potential_module(x)

    def run_metric_center_calculation(self, verbose: bool = True):
        self.metric_module.calculate_centers(verbose=verbose)

    def run_metric_weight_calculation(self, 
                                      key: jax.random.PRNGKey, 
                                      lr: float = 1e-2,
                                      epochs: int = 100,
                                      batch_size: int = 64,
                                      verbose: bool = True):
        
        self.metric_module.calculate_and_set_weights(
            key=key,
            learning_rate=lr,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )

    def path(self, x, y, num_points=20):
        assert x.ndim == 1 and y.ndim == 1
        assert x.shape[0] == self.D + self.C
        assert y.shape[0] == self.D + self.C

        condition = x[self.D:]
        init_spline_params = jax.lax.stop_gradient(self.predict_spline_params(x, y))
        initializing = self.is_mutable_collection("params")
        
        if initializing:
            ts = jnp.linspace(0, 1, num_points)
            xs = splines.compute_spline(
                x=x[:self.D],
                y=y[:self.D],
                basis=self.spline_amortizer.basis,
                params=init_spline_params,
                ts=ts,
            )
            condition_repeated = jnp.tile(condition.reshape(1, -1), (num_points, 1))
            xs = jnp.concatenate([xs, condition_repeated], axis=-1)
            return xs

        out = self.spline_geodesic_solver.solve(
            self.curve_energy,
            x,
            y,
            init_params=init_spline_params,
            num_final_points=num_points,
        )
        return out.mu

    def energy_at_point(self, x, v):
        M = self.metric(x)
        v = v[:self.D]
        if (
            self.distance_mode == DistanceModes.GEODESIC
            or self.distance_mode == DistanceModes.SQUARED_GEODESIC
        ):
            kinetic = jnp.sqrt(v @ M @ v)
        else:
            kinetic = 0.5 * v @ M @ v

        lagrangian_potential = (self.lagrangian_potential(x) if self.lagrangian_potential_initializer_fn is not None else 0.0)

        return kinetic - lagrangian_potential

    def curve_energy(self, xs):
        assert xs.ndim == 2
        ds = (xs[1:] - xs[:-1]) + 1e-6
        Es = jax.vmap(self.energy_at_point)(xs[:-1], ds)
        return Es.sum()

    def cost(self, x, y):
        assert x.ndim == 1 and y.ndim == 1
        gamma = self.path(x, y)
        E = self.curve_energy(gamma)
        if self.distance_mode == DistanceModes.SQUARED_GEODESIC:
            E = 0.5 * E**2
        return E

    def project(self, x):
        x_ambient = x[:self.D]
        clipped_x = jnp.clip(x_ambient, *self.bounds)

        #add back the condition
        if self.C > 0:
            condition = x[self.D:]
            return jnp.concatenate([clipped_x, condition], axis=-1)
        else:
            return clipped_x
        
    #project for a batch of points
    def batch_project(self, x):
        x_ambient = x[:, :self.D]
        clipped_x = jnp.clip(x_ambient, *self.bounds)

        #add back the condition
        if self.C > 0:
            condition = x[:, self.D:]
            return jnp.concatenate([clipped_x, condition], axis=-1)
        else:
            return clipped_x
    
    def get_ambient_dims(self, x):
        assert x.ndim == 2
        assert x.shape[1] == self.D + self.C
        return x[:, :self.D]

    
    def point_on_path(self, x: jax.numpy.ndarray, y: jax.numpy.ndarray, t_fraction: float) -> jax.numpy.ndarray:
        """
        Computes a point on the spline-based geodesic path between x and y
        at a given time t_fraction.

        Args:
            x: Starting point (D+C dimensional).
            y: Ending point (D+C dimensional).
            t_fraction: Fractional time, float between 0.0 and 1.0.

        Returns:
            The interpolated point (D+C dimensional).
        """
        assert x.ndim == 1 and y.ndim == 1, "Input points must be 1D arrays."
        assert x.shape == y.shape, "Input points must have the same shape."
        assert x.shape[0] == self.D + self.C, "Input points must have shape (D + C,)."

        x_spatial = x[:self.D]
        y_spatial = y[:self.D]

        spline_p = self.predict_spline_params(x, y) 

        interpolated_spatial_array = splines.compute_spline(
            x=x_spatial,
            y=y_spatial,
            basis=self.spline_amortizer.basis, 
            params=spline_p,
            ts=jnp.array([t_fraction])
        )
        interpolated_spatial = interpolated_spatial_array[0]

        if self.C:
            condition_part = x[self.D:]
            interpolated_point = jnp.concatenate([interpolated_spatial, condition_part])
        else:
            interpolated_point = interpolated_spatial
            
        return interpolated_point

    def add_plot_background(self, params, axs, xlims, ylims=None, alpha=1.0, condition=None):
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])

        if issubclass(
            self.metric_initializer_fn, metrics.NeuralNetMetric
            ):
            grid_size = 21

            assert len(xlims) == 2
            if ylims is None:
                ylims = xlims

            if not hasattr(self, "eigs_vmap_jit"):

                @functools.partial(jax.vmap, in_axes=(None, 0))
                def eigs_vmap(params, x):
                    A = self.apply({"params": params}, x, method=self.metric)
                    vals, vecs = jnp.linalg.eigh(A)
                    return vals, vecs.T 

                self.eigs_vmap_jit = jax.jit(eigs_vmap)

            xflat, x1, x2 = _get_grid(xlims, ylims, grid_size)

            if self.C:
                if condition is not None:
                    default_condition = jnp.tile(condition, (xflat.shape[0], 1))
                else:
                    default_condition = jnp.zeros((xflat.shape[0], self.C))
                x_eval = jnp.concatenate([xflat, default_condition], axis=-1)
            else:
                x_eval = xflat

            vals, vecs = self.eigs_vmap_jit(params, x_eval)

            # Use only the first 2 components of eigenvectors for 2D quiver plot
            u = vecs[:, 0, 0].reshape(x1.shape)
            v = vecs[:, 0, 1].reshape(x1.shape)

            for ax in axs:
                ax.quiver(x1.ravel(), x2.ravel(), u, v, alpha=alpha)
                ax.quiver(x1.ravel(), x2.ravel(), -u, -v, alpha=alpha)

                ax.set_xlim(*xlims)
                ax.set_ylim(*ylims)
        if self.lagrangian_potential_initializer_fn is not None:
            grid_size = 201
            assert len(xlims) == 2
            if ylims is None:
                ylims = xlims

            if not hasattr(self, "lagrangian_potential_vmap_jit"):

                @functools.partial(jax.vmap, in_axes=(None, 0))
                def lagrangian_potential_vmap(params, x):
                    return self.apply(
                        {"params": params}, x, method=self.lagrangian_potential
                    )

                self.lagrangian_potential_vmap_jit = jax.jit(lagrangian_potential_vmap)

            xflat, x1, x2 = _get_grid(xlims, ylims, grid_size)

            if self.C:
                if condition is not None:
                    default_condition = jnp.tile(condition, (xflat.shape[0], 1))
                else:
                    default_condition = jnp.zeros((xflat.shape[0], self.C))
                x_eval = jnp.concatenate([xflat, default_condition], axis=-1)
            else:
                x_eval = xflat

            for ax in axs:
                ax.set_xlim(*xlims)
                ax.set_ylim(*ylims)

    def __hash__(self):
        return hash(
            (
                self.distance_mode,
                self.metric_initializer_fn,
                self.spline_model_initializer_fn,
                self.lagrangian_potential_initializer_fn,
            )
        )


def _get_grid(xlims: Tuple[float, float], ylims: Tuple[float, float], grid_size=21):
    xs = np.linspace(*xlims, num=grid_size)
    ys = np.linspace(*ylims, num=grid_size)
    x1, x2 = np.meshgrid(xs, ys)
    x1flat = x1.ravel()
    x2flat = x2.ravel()
    xflat = np.stack((x1flat, x2flat)).T
    return xflat, x1, x2