import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional
from dataclasses import dataclass
from abc import abstractmethod


@dataclass
class MetricBase(nn.Module):
    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError


@dataclass
class EuclideanMetric(MetricBase):
    D: int = 2
    C: int = 0
    categorical: Optional[bool] = False
    num_categories: Optional[int] = 4

    def __call__(self, x):
        assert x.ndim == 1
        return jnp.eye(self.D)

@dataclass
class NeuralNetMetric(MetricBase):
    D: int = 2
    C: int = 0
    hidden_dim: int = 128
    categorical: Optional[bool] = False
    num_categories: Optional[int] = 4
    use_film: Optional[bool] = True 

    @nn.compact
    def __call__(self, x):
        assert x.ndim == 1
        assert self.D == 2

        if self.categorical:
            assert x.shape[0] == self.D + 1, f"Expected categorical input shape ({self.D + 1},), got {x.shape}"
            x_ambient = x[:self.D]
            category_index = x[self.D].astype(jnp.int32)
            embedding_dim = max(16, self.hidden_dim // 4)
            cat_embedding = nn.Embed(num_embeddings=self.num_categories, features=embedding_dim, name="category_embedding")(category_index)

            h_spatial = nn.Dense(self.hidden_dim, name="spatial_dense_0")(x_ambient)

            film_hidden_dim = max(16, self.hidden_dim // 4)
            film_params = nn.Dense(film_hidden_dim, name="cat_film_dense_0")(cat_embedding)
            film_params = nn.leaky_relu(film_params)
            film_params = nn.Dense(2 * self.hidden_dim, name="cat_film_dense_1")(film_params)

            gamma = film_params[:self.hidden_dim]
            beta = film_params[self.hidden_dim:]

            h = gamma * h_spatial + beta
            h = nn.leaky_relu(h)
            current_hidden_idx = 1

        elif self.C > 0 and self.use_film and not self.categorical:
            assert x.shape[0] == self.D + self.C, f"Expected continuous conditional input shape ({self.D + self.C},), got {x.shape}"
            x_ambient = x[:self.D]
            c = x[self.D:]

            # --- FiLM ---
            h_spatial = nn.Dense(self.hidden_dim, name="spatial_dense_0")(x_ambient)
            film_hidden_dim = max(16, self.hidden_dim // 4)
            film_params = nn.Dense(film_hidden_dim, name="film_dense_0")(c)
            film_params = nn.leaky_relu(film_params)
            # Output size is 2 * target activation size (gamma and beta)
            film_params = nn.Dense(2 * self.hidden_dim, name="film_dense_1")(film_params)

            gamma = film_params[:self.hidden_dim]
            beta = film_params[self.hidden_dim:]

            # Apply FiLM
            h = gamma * h_spatial + beta
            h = nn.leaky_relu(h)
            # --- End FiLM ---
            current_hidden_idx = 1

        else: #just simple concatenation for condition
            assert x.shape[0] == self.D + self.C, f"Expected concatenated input shape ({self.D + self.C},), got {x.shape}"
            net_input = x
            h = nn.Dense(self.hidden_dim, name="dense_0")(net_input)
            h = nn.leaky_relu(h)
            current_hidden_idx = 1


        h = nn.Dense(self.hidden_dim, name=f"dense_{current_hidden_idx}")(h)
        h = nn.leaky_relu(h)
        nn_out = nn.Dense(2, name="output_dense")(h)

        theta = jnp.arctan2(*nn_out.squeeze())
        R = jnp.array([[jnp.cos(theta), -jnp.sin(theta)],
                       [jnp.sin(theta), jnp.cos(theta)]])
        Q = jnp.array([[1., 0.],
                       [0., 0.1]])
        A = R.T @ Q @ R
        return A


@dataclass
class NeuralNetMetricEig(MetricBase):
    D: int = 2 
    C: int = 0 
    hidden_dim: int = 128
    categorical: Optional[bool] = False
    num_categories: Optional[int] = 4
    min_eigenvalue: float = 0.1
    max_eigenvalue: float = 1
    temperature: float = 1.0
    total_budget: Optional[float] = None  # Total eigenvalue budget, defaults to D if None
    use_film: Optional[bool] = True # Add a flag to enable/disable FiLM

    @nn.compact
    def __call__(self, x):
        assert x.ndim == 1

        if self.D == 2:
            output_size = self.D + 2
        else:
            output_size = self.D + (self.D * (self.D - 1)) # // 2

        if self.categorical:
            assert x.shape[0] == self.D + 1, f"Expected categorical input shape ({self.D + 1},), got {x.shape}"
            x_ambient = x[:self.D]
            category_index = x[self.D].astype(jnp.int32)
            embedding_dim = max(16, self.hidden_dim // 4)
            cat_embedding = nn.Embed(num_embeddings=self.num_categories, features=embedding_dim, name="category_embedding")(category_index)

            h_spatial = nn.Dense(self.hidden_dim, name="spatial_dense_0")(x_ambient)

            film_hidden_dim = max(16, self.hidden_dim // 4)
            film_params = nn.Dense(film_hidden_dim, name="cat_film_dense_0")(cat_embedding)
            film_params = nn.leaky_relu(film_params)
            film_params = nn.Dense(2 * self.hidden_dim, name="cat_film_dense_1")(film_params)

            gamma = film_params[:self.hidden_dim]
            beta = film_params[self.hidden_dim:]

            h = gamma * h_spatial + beta
            h = nn.leaky_relu(h)
            current_hidden_idx = 1

        elif self.C > 0 and self.use_film and not self.categorical:
            assert x.shape[0] == self.D + self.C, f"Expected continuous conditional input shape ({self.D + self.C},), got {x.shape}"
            x_ambient = x[:self.D]
            c = x[self.D:]

            h_spatial = nn.Dense(self.hidden_dim, name="spatial_dense_0")(x_ambient)
            film_hidden_dim = max(16, self.hidden_dim // 4)
            film_params = nn.Dense(film_hidden_dim, name="film_dense_0")(c)
            film_params = nn.leaky_relu(film_params)
            film_params = nn.Dense(2 * self.hidden_dim, name="film_dense_1")(film_params)

            gamma = film_params[:self.hidden_dim]
            beta = film_params[self.hidden_dim:]

            h = gamma * h_spatial + beta
            h = nn.leaky_relu(h)
            current_hidden_idx = 1

        else: 
            assert x.shape[0] == self.D + self.C, f"Expected concatenated input shape ({self.D + self.C},), got {x.shape}"
            net_input = x
            h = nn.Dense(self.hidden_dim, name="dense_0")(net_input)
            h = nn.leaky_relu(h)
            current_hidden_idx = 1


        h = nn.Dense(self.hidden_dim, name=f"dense_{current_hidden_idx}")(h)
        h = nn.leaky_relu(h)
        nn_out = nn.Dense(output_size, name="output_dense")(h)


        raw_eigenvalues = nn_out[:self.D]
        eigenvalue_weights = jax.nn.softmax(raw_eigenvalues * self.temperature)

        # Set total budget to D if not specified
        budget = self.D if self.total_budget is None else self.total_budget

        # Compute eigenvalues: ensure minimum values while allocating the budget
        # Ensure budget_range calculation avoids issues if max=min
        budget_range = jnp.maximum(0.0, self.max_eigenvalue - self.min_eigenvalue)
        # Scale weights by budget relative to default (D) and the available range
        scaled_weights = eigenvalue_weights * (budget / self.D) * budget_range
        eigenvalues = self.min_eigenvalue + scaled_weights
        eigenvalues = jnp.clip(eigenvalues, self.min_eigenvalue, self.max_eigenvalue)
        rotation = self._create_rotation_matrix(nn_out[self.D:])
        diagonal = jnp.diag(eigenvalues)
        A = rotation.T @ diagonal @ rotation

        return A

    def _create_rotation_matrix(self, params):
        """
        Create a rotation matrix from parameters.
        For D=2: uses arctan2 from 2D vector
        For D>3: D(D-1) parameters for generalized rotation
        """
        if self.D == 2:
            theta = jnp.arctan2(params[1], params[0])
            rotation = jnp.array([
                [jnp.cos(theta), -jnp.sin(theta)],
                [jnp.sin(theta), jnp.cos(theta)]
            ])

        # more stable parametrisation, using 2D vectors to define each axis
        else:
            rotation = jnp.eye(self.D)
            param_idx = 0

            for i in range(self.D):
                for j in range(i + 1, self.D):
                    # Each Givens angle comes from two params
                    x, y = params[param_idx], params[param_idx + 1]
                    param_idx += 2

                    angle = jnp.arctan2(y, x)

                    # Build Givens rotation in the (i, j) plane
                    givens = jnp.eye(self.D)
                    givens = givens.at[i, i].set(jnp.cos(angle))
                    givens = givens.at[j, j].set(jnp.cos(angle))
                    givens = givens.at[i, j].set(-jnp.sin(angle))
                    givens = givens.at[j, i].set(jnp.sin(angle))

                    rotation = rotation @ givens

        return rotation
    