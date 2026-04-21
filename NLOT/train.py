import functools
import csv
import time
import jax
import jax.numpy as jnp
import numpy as np
import os
import optax
import cloudpickle as pkl
from flax.core import FrozenDict
import hydra
from omegaconf import OmegaConf
from lagrangian_ot import models, neuraldual, geometries, data, lagrangian_potentials
from generate_synth_data import log_likelihood_conditional_semicircle
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
from IPython.core import ultratb
import wandb
import ot

plt.style.use('bmh')
sys.excepthook = ultratb.FormattedTB(
    mode='Plain',
    color_scheme='Neutral',
    call_pdb=bool(int(os.environ.get("HTI_DEBUG_PDB", "0"))),
)

class Workspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.work_dir = os.getcwd()
        print(f"workspace: {self.work_dir}")

        wandb.init(
            project=cfg.get("wandb_project", "NLOT-Scarvelis_" + self.cfg.geometry), 
            config=OmegaConf.to_container(cfg, resolve=True), 
            name=cfg.get("run_name", None),
            mode="disabled" if not cfg.get("wandb", True) else "online",
        )
        self.data = self.cfg.dataset
        dataset_path = self.cfg.get("dataset_path", None)
        if dataset_path is not None:
            dataset_path = hydra.utils.to_absolute_path(dataset_path)

        self.key = jax.random.PRNGKey(self.cfg.seed)
        self.elapsed_time = 0.
        self.samplers = data.get_samplers(
            self.data,
            num_pairs_requested=self.cfg.get('num_pairs', None),
            dataset_path=dataset_path,
        )
        self.all_samples = jnp.concatenate([next(s) for s in self.samplers], axis=0)
        
        print(f"all_samples shape: {self.all_samples.shape}")

        if self.cfg.get('include_inverse_potential', False):
            if self.cfg.get('categorical', False):
                lagrangian_potential_initializer_fn = lagrangian_potentials.DensityPotentialCircles(
                    D=self.cfg.get('D', 2),
                    C=self.cfg.get('C', 0),
                    samples=self.all_samples,
                    bandwidth=self.cfg.get('bandwidth', 1.0),
                    lambda_repel=self.cfg.get('lambda', 0.01),
                )
            else:
                lagrangian_potential_initializer_fn = lagrangian_potentials.DensityPotential(
                    D=self.cfg.get('D', 2),
                    C=self.cfg.get('C', 0),
                    samples=self.all_samples,
                    bandwidth=self.cfg.get('bandwidth', 1.0),
                    conditional_bandwidth=self.cfg.get('conditional_bandwidth', 1.0),
                    lambda_repel=self.cfg.get('lambda', 0.01),
                )
        else:
            lagrangian_potential_initializer_fn = None

        self.geometry = geometries.get(
            self.cfg.geometry, 
            self.cfg.get('geometry_kwargs', {}),
            D=self.cfg.get('D', 2),
            C=self.cfg.get('C', 0),
            categorical=self.cfg.get('categorical', False),
            num_categories=self.cfg.get('num_categories', 0),
            lagrangian_potential_initializer_fn=lagrangian_potential_initializer_fn
        )

        if 'euclidean' in self.cfg.geometry or 'neural' in self.cfg.geometry or 'land' in self.cfg.geometry:
            if self.data is None:
                raise ValueError('data must be specified for euclidean and neural geometries')

        self.has_reference_geometry = 'neural' in self.cfg.geometry or 'land' in self.cfg.geometry
        if self.has_reference_geometry:
            self.reference_geometry = geometries.get(
                self.cfg.geometry, 
                self.cfg.get('geometry_kwargs', {}),
                D=self.cfg.get('D', 2),
                C=self.cfg.get('C', 0),
                categorical=self.cfg.get('categorical', False),
                num_categories=self.cfg.get('num_categories', 0),
                lagrangian_potential_initializer_fn=lagrangian_potential_initializer_fn
            )

        if self.data is None:
            self.data = self.cfg.geometry

        self.geometry.bounds, self.geometry.xbounds, self.geometry.ybounds = data.get_bounds(
            self.data,
            samples=self.all_samples,
            custom_bounds=self.cfg.get('plot_bounds', None),
        )

        self.num_pairs = len(self.samplers) - 1
        self.time_points = cfg.get('time_points', np.linspace(0, 1, self.num_pairs + 1))

        print(f'training on {self.num_pairs} pairs at times {self.time_points}')
        self.eval_samples = [next(s) for s in self.samplers]
        self.optimizer_target_potential = optax.adamw(learning_rate=self.cfg.potential_lr)
        self.optimizer_source_map = self.optimizer_target_potential
        self.optimizer_geom = optax.adamw(learning_rate=self.cfg.metric.lr)
        k1, self.key = jax.random.split(self.key)
        self.params_geometry = self.geometry.init(
            k1, self.eval_samples[0][0], self.eval_samples[1][0],
            method=self.geometry.cost
        ).get('params', {})
        self.params_geometry = FrozenDict(self.params_geometry)
        self.state_geometry = self.optimizer_geom.init(self.params_geometry)

        target_potential = models.MLP(
            dim_hidden=self.cfg.target_potential_dim_hidden,
            is_potential=True,
            D=self.cfg.get('D', 2),
            C=self.cfg.get('C', 0),
            categorical=self.cfg.get('categorical', False),
            num_categories=self.cfg.get('num_categories', 0),
        )
        
        source_map = models.MLP(
            dim_hidden=self.cfg.source_map_dim_hidden,
            is_potential=False,
            D=self.cfg.get('D', 2),
            C=self.cfg.get('C', 0),
            categorical=self.cfg.get('categorical', False),
            num_categories=self.cfg.get('num_categories', 0),
        )
        
        ctransform_solver = hydra.utils.instantiate(self.cfg.ctransform_solver)

        self.neural_dual_solver = neuraldual.ManifoldW2NeuralDual(
            geometry=self.geometry,
            target_potential=target_potential,
            source_map=source_map,
            ctransform_solver=ctransform_solver,
        )
        
        init_key, self.key = jax.random.split(self.key)
        state_target_potential, state_source_map = self.neural_dual_solver.initialize_states(
            self.optimizer_target_potential, self.optimizer_source_map,
            init_key, self.eval_samples[0], self.eval_samples[1])
        self.state_target_potentials = [state_target_potential]
        self.state_source_maps = [state_source_map]

        if 'spline_model' in self.params_geometry:
            self.fit_spline_amortizer(self.samplers, init=True)

        self.train_step = 0

    def fit_spline_amortizer(self, samplers, init):
        num_iters = self.cfg.spline.init_train_iters if init else self.cfg.spline.train_iters

        if init:
            def sampler(key):
                # sample from random pairs of source and target
                t = 0
                while True:
                    source_samples = next(samplers[t])
                    target_samples = next(samplers[t+1])
                    all_samples = jnp.concatenate([source_samples, target_samples], axis=0)
                    k1, key = jax.random.split(key)
                    all_samples = jax.random.permutation(k1, all_samples)
                    t = (t + 1) % self.num_pairs
                    yield all_samples

            k1, k2, self.key = jax.random.split(self.key, 3)
            xsampler = iter(sampler(k1))
            ysampler = iter(sampler(k2))
        else:
            def xsampler():
                t = 0
                while True:
                    source_samples = next(samplers[t])
                    t = (t + 1) % self.num_pairs
                    yield source_samples

            def ysampler():
                key = jax.random.PRNGKey(0)
                t = 0
                while True:
                    source_samples = next(samplers[t])
                    transported_samples = self.neural_dual_solver.source_map_apply_jit(
                        {'params': self.state_source_maps[t].params}, source_samples)
                    if self.cfg.spline.noise > 0.:
                        k1, key = jax.random.split(key)
                        transported_samples += self.cfg.spline.noise * jax.random.normal(
                            key, transported_samples.shape)
                    t = (t + 1) % self.num_pairs
                    yield transported_samples

            xsampler = iter(xsampler())
            ysampler = iter(ysampler())


        self.params_geometry = self.geometry.spline_amortizer.train(
            self.params_geometry,
            xsampler, ysampler,
            max_iter=num_iters,
            grad_norm_threshold=self.cfg.spline.grad_norm_threshold,
        )

    def update_all_states(self, state_target_potentials, state_source_maps, batches):
        out = []
        for t in range(self.num_pairs):
            out_t = self.neural_dual_solver.update_fn_jit(
                state_target_potentials[t if self.train_step > 0 else 0],
                state_source_maps[t if self.train_step > 0 else 0],
                self.params_geometry,
                batches[t],
            )
            out.append(out_t)

            if self.cfg.spline.update_on_conjugates \
                    and 'spline_model' in self.params_geometry:
                _, info = out_t
                self.params_geometry = self.geometry.spline_amortizer.train_single(
                    self.params_geometry,
                    batches[t]['source'], info.target_hat,
                    verbose=False,
                )

        new_states, infos = zip(*out)
        new_states = zip(*new_states)
        mean_info = type(infos[0])(*[jnp.array(x).mean() for x in list(zip(*infos))])
        return new_states, mean_info

    def sample_all_batches(self, samplers):
        batches = []
        for t in range(self.num_pairs):
            batches.append({
                "source": jnp.asarray(next(samplers[t])),
                "target": jnp.asarray(next(samplers[t+1])),
            })
        return batches

    def geometry_loss(self, params_geometry, state_target_potentials, state_source_maps, batches, key):
        dual_losses = []

        for t in range(self.num_pairs):
            batch = batches[t]
            _, info_t = self.neural_dual_solver.loss_fn(
                state_target_potentials[t].params,
                state_source_maps[t].params,
                params_geometry, 
                batch
            )
            dual_losses.append(-info_t.dual_loss)
            

        mean_dual_loss = jnp.mean(jnp.stack(dual_losses)) 
        
        total_loss = mean_dual_loss

        return total_loss

    @functools.partial(jax.jit, static_argnums=[0])
    def update_geometry(self, params_geometry, state_geometry, state_target_potentials, state_source_maps, batches, key):
        geometry_grad_fn = jax.value_and_grad(self.geometry_loss)
        loss, grads = geometry_grad_fn(
            params_geometry,
            state_target_potentials, 
            state_source_maps,
            batches, 
            key
        )

        updates, new_state_geometry = self.optimizer_geom.update(
            grads,
            state_geometry,
            FrozenDict(params_geometry) # Ensure params are passed as FrozenDict
        )

        new_params_geometry = optax.apply_updates(params_geometry, updates)

        return new_params_geometry, new_state_geometry, loss

    def _get_marginal_eval_data(self):
        """Returns (time_0_points, eval_points) or None when no eval set is configured."""
        base_path = os.path.dirname(os.path.realpath(__file__))

        if self.data == "conditional_semicircles":
            test_data = jnp.load(os.path.join(base_path, "eval_data", "eval_marginals_semicircle.pkl"), allow_pickle=True)
            time_0_points = test_data[0][1]
            return time_0_points, test_data[1:]

        if self.data == "2moons_dropout":
            test_path = os.path.join(base_path, "data", "diffusion_2moons_dropout.pt")
            test_data = torch.load(test_path).cpu().numpy()
            test_data = jnp.array(test_data)
            time_0_points = test_data[0]
            eval_points = [
                (0, test_data[0]), (0.1 / 0.99, test_data[1]), (0.2 / 0.99, test_data[2]),
                (0.3 / 0.99, test_data[3]), (0.4 / 0.99, test_data[4]), (0.6 / 0.99, test_data[6]),
                (0.7 / 0.99, test_data[7]), (0.8 / 0.99, test_data[8]), (0.9 / 0.99, test_data[9]),
            ]
            return time_0_points, eval_points[1:]

        return None

    def run(self):

        logf, writer = self._init_logging()

        while self.train_step < self.cfg.num_train_iters:
            start = time.time()
            batches = self.sample_all_batches(self.samplers)


            new_states, info = self.update_all_states(
                self.state_target_potentials,
                self.state_source_maps,
                batches
            )
            self.state_target_potentials, self.state_source_maps = new_states

            update_step_time = time.time() - start
            self.elapsed_time += update_step_time            

            if self.train_step % self.cfg.metric.update_frequency == 0:
                start = time.time()
                k1, self.key = jax.random.split(self.key)
                new_params_geometry, new_state_geometry, geom_loss = self.update_geometry(
                    self.params_geometry, 
                    self.state_geometry,
                    self.state_target_potentials, 
                    self.state_source_maps,
                    batches, 
                    k1
                )
                self.params_geometry, self.state_geometry = new_params_geometry, new_state_geometry
                update_metric_time = time.time() - start
                self.elapsed_time += update_metric_time
                print(
                    f'step: {self.train_step}/{self.cfg.num_train_iters} '
                    f'dual_loss: {info.dual_loss:.2e}, amor_loss: {info.amor_loss:.2e} '
                    f'geom_loss: {geom_loss:.2e} '
                    f'update_step_time: {update_step_time:.2f}s '
                    f'update_metric_time: {update_metric_time:.2f}s '
                )
                wandb.log({
                    "train/dual_loss": info.dual_loss,
                    "train/amor_loss": info.amor_loss,
                    "train/geom_loss": geom_loss,
                    "train/elapsed_time": self.elapsed_time,
                    "train/mean_potential_target": info.mean_potential_target,
                    "train/min_potential_target": info.min_potential_target,
                    "train/max_potential_target": info.max_potential_target,
                    "train/mean_potential_target_hat": info.mean_potential_target_hat,
                    "train/min_potential_target_hat": info.min_potential_target_hat,
                    "train/max_potential_target_hat": info.max_potential_target_hat,
                }, step=self.train_step)

            else:
                print(
                    f'step: {self.train_step}/{self.cfg.num_train_iters} '
                    f'dual_loss: {info.dual_loss:.2e}, amor_loss: {info.amor_loss:.2e} '
                    f'update_step_time: {update_step_time:.2f}s '
                )
                wandb.log({
                    "train/dual_loss": info.dual_loss,
                    "train/amor_loss": info.amor_loss,
                    "train/elapsed_time": self.elapsed_time,
                    "train/mean_potential_target": info.mean_potential_target,
                    "train/min_potential_target": info.min_potential_target,
                    "train/max_potential_target": info.max_potential_target,
                    "train/mean_potential_target_hat": info.mean_potential_target_hat,
                    "train/min_potential_target_hat": info.min_potential_target_hat,
                    "train/max_potential_target_hat": info.max_potential_target_hat,
                 }, step=self.train_step)


            if self.train_step % self.cfg.spline.update_frequency == 0 and 'spline_model' in self.params_geometry and self.train_step < self.cfg.num_train_iters:
                self.fit_spline_amortizer(samplers=self.samplers, init=False)

            if not self.cfg.plotting.get('disable', False):
                if self.train_step % self.cfg.plot_frequency == 0:
                    self.plot_pushforward()

                    marginal_eval_data = self._get_marginal_eval_data()
                    if marginal_eval_data is not None:
                        time_0_points, eval_points = marginal_eval_data
                        print("Evaluating marginals")
                        self.evaluate_marginals(time_0_points, eval_points)

                    

            writer.writerow({
                'iter': self.train_step,
                'ot_cost': -info.dual_loss,
                'elapsed_time': self.elapsed_time,
            })
            logf.flush()

            self.train_step += 1
            if self.train_step % self.cfg.save_frequency == 0:
                self.save()


    def _init_logging(self):
        logf = open('log.csv', 'a')
        fieldnames = ['iter', 'ot_cost', 'elapsed_time']
        writer = csv.DictWriter(logf, fieldnames=fieldnames)
        if os.stat('log.csv').st_size == 0:
            writer.writeheader()
            logf.flush()
        return logf, writer

    def _setup_ax(self, ax, condition=None):
        """Configure plotting axes consistently across datasets."""
        xbounds = self.geometry.xbounds
        ybounds = self.geometry.ybounds
        if len(xbounds) == 2:
            ax.set_xlim(float(xbounds[0]), float(xbounds[1]))
        if len(ybounds) == 2:
            ax.set_ylim(float(ybounds[0]), float(ybounds[1]))

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if condition is not None:
            cond_values = np.asarray(condition).tolist()
            ax.set_title(f"condition={cond_values}")

    def _clean_axis(self, ax):
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal', adjustable='box')

    def plot_pushforward(self, num_samples=100):
        num_samples = min(num_samples, self.eval_samples[0].shape[0])
        all_init_xs = jax.random.choice(
            jax.random.PRNGKey(0), self.eval_samples[0], shape=(num_samples,), replace=False)

        if self.cfg.C > 0 and self.cfg.categorical:
            condition_vectors = all_init_xs[:, self.cfg.D:]
            unique_conditions = jnp.unique(condition_vectors, axis=0)
            num_conditions = unique_conditions.shape[0]
        else:
            unique_conditions = None 
            num_conditions = 1

        cmap = plt.get_cmap('viridis', num_conditions)
        norm = mpl.colors.Normalize(vmin=0, vmax=num_conditions - 1)

        for c_idx in range(num_conditions):

            fig, ax = plt.subplots(figsize=(8, 4))
            
            if unique_conditions is not None:
                current_condition = unique_conditions[c_idx]
                condition_matches = jnp.all(condition_vectors == current_condition, axis=1)
                condition_indices = jnp.where(condition_matches)[0]
                self._setup_ax(ax, condition=current_condition)
            else:
                condition_indices = jnp.arange(all_init_xs.shape[0])
                self._setup_ax(ax)

            plot_color = cmap(norm(c_idx))

            init_xs_condition = all_init_xs[condition_indices]

            for i in range(init_xs_condition.shape[0]):
                init_x = init_xs_condition[i]
                ax.scatter([init_x[0]], [init_x[1]], s=20, alpha=1,
                            zorder=10, c=[plot_color])

                x = init_x
                for t in range(self.num_pairs):
                    prev_x = x

                    x = self.neural_dual_solver.pushforward_jit(
                        self.state_source_maps[t].params,
                        self.state_target_potentials[t].params,
                        self.params_geometry,
                        x
                    ).solution

                    path = self.neural_dual_solver.path_jit(
                        self.params_geometry, prev_x, x)

                    ax.plot(
                        path[:, 0], path[:, 1],
                        color=plot_color, 
                        alpha=0.5,
                        lw=3,
                    )

            self._clean_axis(ax)
            fname = f'pushforward_condition_{c_idx}.png' 
            print(f'saving to {fname}')
            fig.savefig(fname, bbox_inches='tight', pad_inches=0)
            wandb.log({f"plots/pushforward_condition_{c_idx}": wandb.Image(fname)}, step=self.train_step)
            plt.close(fig)

    def save(self, tag="latest"):
        path = os.path.join(self.work_dir, f"{tag}.pkl")
        print(f"Saving to {path}")

        collect_path = None
        collect_dir = self.cfg.get("collect_save_dir", None)
        if collect_dir:
            os.makedirs(collect_dir, exist_ok=True)
            collect_fname = (
                f"{self.cfg.dataset}_{self.cfg.geometry}_{self.cfg.include_inverse_potential}"
                f"_{self.cfg.seed}_{self.cfg.get('lambda', 0)}.pkl"
            )
            collect_path = os.path.join(collect_dir, collect_fname)
            print(f"Saving collection copy to {collect_path}")

        # Temporarily remove non-picklable samplers
        samplers_backup = self.samplers
        self.samplers = None

        try:
            with open(path, "wb") as f:
                pkl.dump(self, f)
            if collect_path is not None:
                with open(collect_path, "wb") as f:
                    pkl.dump(self, f)
        finally:
            # Restore samplers
            self.samplers = samplers_backup

    def predictor_map_for_assignment(self, x_batch, t = 0):
        
        params_source_map = self.state_source_maps[t].params

        return self.neural_dual_solver.source_map_apply_jit(
            {'params': params_source_map},
            x_batch
        )

    def compute_wasserstein_distance(self, samples1, samples2):
        """
        Compute Wasserstein distance between two sets of samples using POT library.
        
        Args:
            samples1: JAX array of shape (n_samples1, dim)
            samples2: JAX array of shape (n_samples2, dim)
            
        Returns:
            float: The Wasserstein distance
        """
        # Convert from JAX arrays to numpy for POT library compatibility
        samples1_np = np.array(samples1)
        samples2_np = np.array(samples2)
        

        M = ot.dist(samples1_np, samples2_np)
        a = np.ones(samples1_np.shape[0]) / samples1_np.shape[0]
        b = np.ones(samples2_np.shape[0]) / samples2_np.shape[0]  
            
        return float(ot.emd2(a, b, M))
        
    def evaluate_marginals(self, initial_samples_at_t0, evaluation_points):
        """
        Evaluates model by transporting initial samples through learned maps and
        geodesic paths, comparing with ground truth time marginals.

        For each interval [T_k, T_{k+1}], samples are transported from T_k.
        If an `eval_time` falls within this interval:
        - At T_k or T_{k+1}: uses directly transported samples.
        - Between T_k and T_{k+1}: interpolates along the geodesic path for time (eval_time - T_k) / (T_{k+1} - T_k).

        Args:
            initial_samples_at_t0 (jax.numpy.ndarray): Samples at self.time_points[0].
            evaluation_points (list[tuple[float, jax.numpy.ndarray]]):
                List of (eval_time, ground_truth_samples_at_eval_time).
        Returns:
            dict: Discrepancy metrics for each evaluation point.
        """
        evaluation_points.sort(key=lambda x: x[0])
        min_train_time = self.time_points[0]
        max_train_time = self.time_points[self.num_pairs]

        valid_evaluation_points = []
        for t_eval, samples in evaluation_points:
            if not (min_train_time <= t_eval <= max_train_time):
                print(f"Warning: Evaluation time {t_eval:.4f} is outside the trained range "
                      f"[{min_train_time:.4f}, {max_train_time:.4f}]. Skipping.")
                continue
            valid_evaluation_points.append((t_eval, samples))
        
        evaluation_points = valid_evaluation_points
        
        metrics_log = {}
        current_transported_samples = initial_samples_at_t0
        eval_point_idx = 0

        is_semicircle_eval = self.data == "conditional_semicircles"
        is_dropout_eval = self.data == "2moons_dropout"

        all_wasserstein_distances = []
        all_circle_distances = []
        all_log_likelihoods = []
        for k in range(self.num_pairs):
            T_k = self.time_points[k]
            T_k_plus_1 = self.time_points[k+1]

            params_source_map_k = self.state_source_maps[k].params
            end_samples_pred_at_Tk_plus_1 = self.neural_dual_solver.source_map_apply_jit(
                {'params': params_source_map_k},
                current_transported_samples
            )
            end_samples_pred_at_Tk_plus_1 = self.geometry.batch_project(end_samples_pred_at_Tk_plus_1)

            @jax.jit
            def interpolate_batch_in_interval(current_geom_params, start_samples_batch, end_samples_pred_for_batch, s_fraction_val):
                return jax.vmap(
                    lambda x_start, y_end, s_f: self.geometry.apply(
                        {'params': current_geom_params},
                        x_start,
                        y_end,
                        s_f,
                        method=self.geometry.point_on_path
                    ), 
                    in_axes=(0, 0, None)
                )(start_samples_batch, end_samples_pred_for_batch, s_fraction_val)

            while eval_point_idx < len(evaluation_points):
                eval_time, true_eval_samples = evaluation_points[eval_point_idx]
                
                if eval_time > T_k_plus_1:
                    break 
                
                predicted_eval_samples = None

                if np.isclose(eval_time, T_k):
                    predicted_eval_samples = current_transported_samples
                elif np.isclose(eval_time, T_k_plus_1):
                    predicted_eval_samples = end_samples_pred_at_Tk_plus_1
                else:
                    s_fraction = (eval_time - T_k) / (T_k_plus_1 - T_k)
                    predicted_eval_samples = interpolate_batch_in_interval(
                        self.params_geometry,
                        current_transported_samples,
                        end_samples_pred_at_Tk_plus_1,
                        s_fraction
                    )
                
                predicted_eval_samples = self.geometry.batch_project(predicted_eval_samples)
                predicted_spatial_overall = predicted_eval_samples[:, :self.cfg.D]
                actual_spatial_overall = true_eval_samples[:, :self.cfg.D]

                if is_dropout_eval:
                    # Generative-dropout evaluation: Wasserstein only.
                    if self.cfg.C:
                        true_conditions_all = true_eval_samples[:, self.cfg.D:]
                        predicted_conditions_all = predicted_eval_samples[:, self.cfg.D:]
                        unique_true_conditions = jnp.unique(true_conditions_all, axis=0)

                        per_condition_wasserstein = []
                        for cond_idx, true_cond_vec in enumerate(unique_true_conditions):
                            true_cond_mask = jnp.all(true_conditions_all == true_cond_vec, axis=1)
                            pred_cond_mask = jnp.all(predicted_conditions_all == true_cond_vec, axis=1)

                            true_samples_for_cond = true_eval_samples[true_cond_mask][:, :self.cfg.D]
                            pred_samples_for_cond = predicted_eval_samples[pred_cond_mask][:, :self.cfg.D]

                            if true_samples_for_cond.shape[0] > 0 and pred_samples_for_cond.shape[0] > 0:
                                wasserstein_dist = self.compute_wasserstein_distance(pred_samples_for_cond, true_samples_for_cond)
                                per_condition_wasserstein.append(wasserstein_dist)
                                metrics_log[f"time_{eval_time:.4f}_cond_{cond_idx}_wass"] = float(wasserstein_dist)

                        if per_condition_wasserstein:
                            avg_wasserstein = jnp.mean(jnp.array(per_condition_wasserstein))
                            metrics_log[f"time_{eval_time:.4f}_avg_wass"] = float(avg_wasserstein)
                            all_wasserstein_distances.extend(per_condition_wasserstein)
                    else:
                        wasserstein_dist = self.compute_wasserstein_distance(predicted_spatial_overall, actual_spatial_overall)
                        metrics_log[f"time_{eval_time:.4f}_wass"] = float(wasserstein_dist)
                        all_wasserstein_distances.append(wasserstein_dist)

                # Circle distance
                if is_semicircle_eval and self.cfg.C > 0:
                    true_conditions_all = true_eval_samples[:, self.cfg.D:]
                    predicted_conditions_all = predicted_eval_samples[:, self.cfg.D:]
                    unique_true_conditions = jnp.unique(true_conditions_all, axis=0)
                    circle_centers = {0: jnp.array([-1.0, 0.0]), 1: jnp.array([-1.0, 0.0]), 2: jnp.array([1.0, 0.0]), 3: jnp.array([1.0, 0.0])}
                    circle_radius = 1.0
                    per_condition_circle_dist = []

                    for cond_idx, true_cond_vec in enumerate(unique_true_conditions):
                        true_cond_mask = jnp.all(true_conditions_all == true_cond_vec, axis=1)
                        pred_cond_mask = jnp.all(predicted_conditions_all == true_cond_vec, axis=1)

                        pred_samples_for_cond = predicted_spatial_overall[pred_cond_mask]

                        if pred_samples_for_cond.shape[0] > 0:
                            center = circle_centers[cond_idx]
                            distances = jnp.abs(jnp.linalg.norm(pred_samples_for_cond - center, axis=1) - circle_radius)
                            avg_distance = jnp.mean(distances)
                            per_condition_circle_dist.append(avg_distance)
                            metrics_log[f"time_{eval_time:.4f}_cond_{cond_idx}_circle_dist"] = float(avg_distance)

                    if per_condition_circle_dist:
                        avg_circle_dist = jnp.mean(jnp.array(per_condition_circle_dist))
                        metrics_log[f"time_{eval_time:.4f}_avg_circle_dist"] = float(avg_circle_dist)
                        all_circle_distances.extend(per_condition_circle_dist)

                if is_semicircle_eval and self.cfg.C > 0:
                    if predicted_eval_samples.shape[1] == self.cfg.D + self.cfg.C:
                        predicted_data_torch = torch.from_numpy(np.array(predicted_eval_samples))
                        log_likelihood_val = -log_likelihood_conditional_semicircle(
                            data=predicted_data_torch,
                            time=eval_time
                        ) / predicted_data_torch.shape[0]
                        metrics_log[f"time_{eval_time:.4f}_log_likelihood"] = float(log_likelihood_val)
                        all_log_likelihoods.append(log_likelihood_val)

                    else:
                        print(f"Skipping log-likelihood for semicircles at t={eval_time:.4f}: predicted_eval_samples shape {predicted_eval_samples.shape} incorrect for conditional data.")

                eval_point_idx += 1

            current_transported_samples = end_samples_pred_at_Tk_plus_1

        if all_wasserstein_distances:
            overall_avg_wasserstein = jnp.mean(jnp.array(all_wasserstein_distances))
            metrics_log["overall_avg_wass"] = float(overall_avg_wasserstein)

        if all_circle_distances:
            overall_avg_circle_dist = jnp.mean(jnp.array(all_circle_distances))
            metrics_log["overall_avg_circle_dist"] = float(overall_avg_circle_dist)

        if all_log_likelihoods:
            overall_avg_log_likelihood = np.mean(np.array(all_log_likelihoods))
            metrics_log["overall_avg_log_likelihood"] = float(overall_avg_log_likelihood)

        if wandb.run is not None and metrics_log:
            wandb.log({"test": metrics_log}, step=self.train_step)

        return metrics_log



@hydra.main(config_path=".", config_name="train.yaml", version_base="1.1")
def main(cfg):
    fname = os.getcwd() + "/latest.pkl"
    if os.path.exists(fname):
        print(f"Resuming from {fname}")
        with open(fname, "rb") as f:
            workspace = pkl.load(f)
        dataset_name = workspace.cfg.get("dataset", workspace.cfg.get("data", None))
        if dataset_name is None:
            raise ValueError("Could not determine dataset name from checkpoint config.")
        workspace.samplers = data.get_samplers(
            dataset_name,
            num_pairs_requested=workspace.cfg.get("num_pairs", None),
            dataset_path=hydra.utils.to_absolute_path(workspace.cfg.get("dataset_path", None))
            if workspace.cfg.get("dataset_path", None) is not None
            else None,
        )
        workspace.data = dataset_name
        print(f"Re-initialized samplers for loaded workspace (num_pairs={len(workspace.samplers)-1})")
    else:
        workspace = Workspace(cfg)

    workspace.run()

if __name__ == '__main__':
    main()
