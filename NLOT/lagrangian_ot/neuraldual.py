from typing import (
    Callable,
    Dict,
    List,
    Literal,
    Optional,
)
import functools
import jax
import jax.numpy as jnp
import collections
import flax
from flax.training import train_state
from lagrangian_ot import ctransform_solvers, models, geometries

Train_t = Dict[Literal["train_logs", "valid_logs"], Dict[str, List[float]]]
Callback_t = Callable
Conj_t = Optional[ctransform_solvers.CTransformSolver]
Potential_t = Callable[[jnp.ndarray], float]

Info = collections.namedtuple("Info", "dual_loss amor_loss num_ctransform_iter target_hat mean_potential_target min_potential_target max_potential_target mean_potential_target_hat min_potential_target_hat max_potential_target_hat")
UpdateOut = collections.namedtuple("UpdateOut", "states info")


class ManifoldW2NeuralDual:
    def __init__(
            self,
            geometry: geometries.GeometryBase,
            target_potential: models.ModelBase,
            source_map: models.ModelBase,
            ctransform_solver: Conj_t = ctransform_solvers.DEFAULT_CTRANSFORM_SOLVER,
            amortization_loss: Literal["objective", "regression"] = "regression",
    ):
        self.geometry = geometry
        self.amortization_loss = amortization_loss

        self.ctransform_solver = ctransform_solver
        self.ctransform_solver.geometry = geometry

        self.target_potential = target_potential
        self.source_map = source_map

        self.target_potential_apply_jit = jax.jit(self.target_potential.apply)
        self.source_map_apply_jit = jax.jit(self.source_map.apply)

        self.finetune_target_hat_vmap_jit = jax.jit(jax.vmap(
            lambda params_target, params_geometry, source, target_init: self.ctransform_solver.solve(
                params_geometry,
                functools.partial(self.target_potential.apply, {'params': params_target}),
                source, 
                target_init=target_init
            ), in_axes=(None, None, 0, 0)))

        self.pushforward = lambda params_source, params_target, params_geometry, source: self.ctransform_solver.solve(
            params_geometry,
            functools.partial(self.target_potential.apply, {'params': params_target}),
            source,
            target_init=self.source_map.apply({'params': params_source}, source)
        )
        self.pushforward_jit = jax.jit(self.pushforward)
        self.pushforward_jit_vmap = jax.jit(jax.vmap(self.pushforward, in_axes=(None, None, None, 0)))

        self.path_jit = jax.jit(
            lambda params_geometry, x, y, num_points=20: self.geometry.apply(
                {'params': params_geometry}, x, y, num_points=num_points, method=self.geometry.path),
            static_argnums=(3,))
        self.path_jit_vmap = jax.jit(jax.vmap(
            lambda params_geometry, x, y, num_points=20: self.geometry.apply(
                {'params': params_geometry}, x, y, num_points=num_points, method=self.geometry.path),
            in_axes=(None, 0, 0, None)), 
            static_argnums=(3,))

        self.update_fn_jit = jax.jit(self.update_fn)

    def initialize_states(self, optimizer_target_potential, optimizer_source_map, key, source_samples, target_samples):
        key, key1, key2 = jax.random.split(key, 3)
        params_target_potential = self.target_potential.init(key1, target_samples)['params']
        params_source_map = self.source_map.init(key2, source_samples)['params']

        state_target_potential = train_state.TrainState.create(
            apply_fn=self.target_potential.apply,
            params=params_target_potential, 
            tx=optimizer_target_potential
        )
        state_source_map = train_state.TrainState.create(
            apply_fn=self.source_map.apply,
            params=params_source_map, 
            tx=optimizer_source_map,
        )
        return state_target_potential, state_source_map


    def state_from_dicts(self, optimizer, net, state_dicts):
        params = state_dicts['params']
        state = train_state.TrainState.create(
            apply_fn=net.apply, 
            params=params, 
            tx=optimizer
        )
        state = flax.serialization.from_state_dict(state, state_dicts)
        return state


    def state_to_dicts(self, state):
        state_dict = flax.serialization.to_state_dict(state)
        return state_dict


    def loss_fn(self, params_target_potential, params_source_map, params_geometry, batch):
        """Loss function for both potentials."""
        source, target = batch["source"], batch["target"]

        init_target_hat = self.source_map.apply({'params': params_source_map}, source)

        target_potential_partial = functools.partial(
            self.target_potential.apply, {'params': params_target_potential})
        if self.ctransform_solver is not None:
            finetune_target_hat = lambda source, target_init: self.ctransform_solver.solve(
                params_geometry, 
                target_potential_partial, 
                source, 
                target_init=target_init
            )
            finetune_target_hat = jax.vmap(finetune_target_hat)
            out = finetune_target_hat(source, init_target_hat)
            target_hat_detach = jax.lax.stop_gradient(out.solution)
            num_ctransform_iter = jnp.mean(out.num_iter)
            target_hat_detach = self.geometry.batch_project(target_hat_detach)
        else:
            target_hat_detach = init_target_hat
            num_ctransform_iter = 0

        # Potential evaluated at actual target samples
        potential_at_target = target_potential_partial(target)
        mean_pot_target = jnp.mean(potential_at_target)
        min_pot_target = jnp.min(potential_at_target)
        max_pot_target = jnp.max(potential_at_target)

        # Potential evaluated at c-transform of source samples
        potential_at_target_hat = target_potential_partial(target_hat_detach)
        mean_pot_target_hat = jnp.mean(potential_at_target_hat)
        min_pot_target_hat = jnp.min(potential_at_target_hat)
        max_pot_target_hat = jnp.max(potential_at_target_hat)

        target_potential = target_potential_partial(target)
        cost_vmap = jax.vmap(lambda x, y: self.geometry.apply(
            {'params': params_geometry}, x, y, method=self.geometry.cost))
        source_potential = cost_vmap(source, target_hat_detach) - target_potential_partial(target_hat_detach)
        dual_source = source_potential.mean()
        dual_target = target_potential.mean()
        dual_loss = -dual_source - dual_target

        if self.amortization_loss == "regression":
            amor_loss = ((self.geometry.get_ambient_dims(init_target_hat) -
                          self.geometry.get_ambient_dims(target_hat_detach)) ** 2).mean()
        else:
            raise ValueError("Amortization loss has been misspecified.")

        loss = dual_loss + amor_loss
        return loss, Info(dual_loss, amor_loss, num_ctransform_iter, target_hat_detach, mean_pot_target, min_pot_target, max_pot_target, mean_pot_target_hat, min_pot_target_hat, max_pot_target_hat)


    def update_fn(self, state_target_potential, state_source_map, params_geometry, batch):
        """Step function of either training or validation."""
        grad_fn = jax.value_and_grad(self.loss_fn, argnums=[0, 1], has_aux=True)
        (loss, info), (grads_target_potential, grads_source_map) = grad_fn(
                state_target_potential.params,
                state_source_map.params,
                params_geometry,
                batch,
        )
        new_states = (
            state_target_potential.apply_gradients(grads=grads_target_potential),
            state_source_map.apply_gradients(grads=grads_source_map)
        )
        return UpdateOut(new_states, info)
