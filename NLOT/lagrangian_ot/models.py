import abc
from typing import Optional, Sequence
import flax.linen as nn
import jax.numpy as jnp


class ModelBase(abc.ABC, nn.Module):
    """Base class for the neural solver models."""

    @property
    @abc.abstractmethod
    def is_potential(self) -> bool:
        """Indicates if the module defines the potential's value or the gradient.

        Returns:
            ``True`` if the module defines the potential's value, ``False``
            if it defines the gradient.
        """

class MLP(ModelBase):
    """A non-convex MLP.

    Args:
        dim_hidden: sequence specifying size of hidden dimensions. The output
            dimension of the last layer is automatically set to 1 if
            :attr:`is_potential` is ``True``, or the dimension of the input otherwise
        is_potential: Model the potential if ``True``, otherwise
            model the gradient of the potential
    """

    dim_hidden: Sequence[int]
    is_potential: bool = True
    D: int = 2
    C: int = 0
    categorical: Optional[bool] = False 
    num_categories: Optional[int] = 4 
    use_film: Optional[bool] = True # Add a flag to enable/disable FiLM


    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        squeeze = x.ndim == 1
        if squeeze:
            x = jnp.expand_dims(x, 0)
        assert x.ndim == 2, x.ndim
        assert x.shape[1] == self.D + self.C

        x_ambient = x[:, :self.D]

        if self.C > 0 and self.use_film and not self.categorical:
            conditional_features = x[:, self.D:]
            h_spatial = nn.Dense(self.dim_hidden[0], name="spatial_dense_0")(x_ambient)

            film_hidden_dim = max(16, self.dim_hidden[0] // 4) # Example size
            film_params = nn.Dense(film_hidden_dim, name="film_dense_0")(conditional_features)
            film_params = nn.relu(film_params)
            # Output size is 2 * target activation size (gamma and beta)
            film_params = nn.Dense(2 * self.dim_hidden[0], name="film_dense_1")(film_params)

            gamma = film_params[:, :self.dim_hidden[0]]
            beta = film_params[:, self.dim_hidden[0]:]

            # Apply FiLM
            # Ensure gamma/beta shapes match h_spatial
            # Add broadcast dimension if needed, e.g., if h_spatial has more dims
            h = gamma * h_spatial + beta
            h = nn.relu(h) # Apply activation after modulation
            current_hidden_idx = 1

        elif self.categorical:
            category_index = x[:, self.D].astype(jnp.int32)
            embedding_dim = max(16, self.dim_hidden[0] // 4)
            cat_embedding = nn.Embed(num_embeddings=self.num_categories, features=embedding_dim, name="category_embedding")(category_index)

            h_spatial = nn.Dense(self.dim_hidden[0], name="spatial_dense_0")(x_ambient)

            film_hidden_dim = max(16, self.dim_hidden[0] // 4)
            film_params = nn.Dense(film_hidden_dim, name="cat_film_dense_0")(cat_embedding)
            film_params = nn.relu(film_params)
            film_params = nn.Dense(2 * self.dim_hidden[0], name="cat_film_dense_1")(film_params)

            gamma = film_params[:, :self.dim_hidden[0]]
            beta = film_params[:, self.dim_hidden[0]:]

            h = gamma * h_spatial + beta
            h = nn.relu(h)
            current_hidden_idx = 1
        else:
            h = nn.Dense(self.dim_hidden[0])(x)
            h = nn.relu(h)
            current_hidden_idx = 1
        
        for i, dim_out in enumerate(self.dim_hidden[current_hidden_idx:]):
            h = nn.Dense(dim_out, name=f"common_dense_{i}")(h)
            h = nn.relu(h)

         # Final output layer
        if self.is_potential:
            output_dim = 1
        else:
            output_dim = self.D

        output = nn.Dense(output_dim, name="output_dense")(h)

        if self.is_potential:
            output = output.squeeze(-1)
        else:
            output = x[:, :self.D] + output
            
            # Add condition to the end
            output = jnp.concatenate([output, x[:, self.D:]], axis=-1)

        return output.squeeze(0) if squeeze else output
