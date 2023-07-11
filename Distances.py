from __future__ import annotations

__all__ = ("Mahalanobis", "Euclidean")

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jax import Array
    from jax.typing import ArrayLike


class Mahalanobis(eqx.Module):
    Vt: Array  # Truncated right singular matrix transposed of the corpus
    corpus_mean: Array  # Mean of the corpus
    S: Array  # Truncated singular values of the corpus
    numerical_rank: int  # Numerical rank
    corpus: ArrayLike  # Corpus used to compute the Data

    def __init__(self, corpus: ArrayLike, svd_threshold: float = 1e-12):
        self.corpus = corpus
        self.corpus_mean = jnp.mean(corpus, axis=0)
        _U, S, Vt = jnp.linalg.svd(corpus - self.corpus_mean)
        self.numerical_rank = int(jnp.sum(svd_threshold <= S))
        self.Vt = Vt[: self.numerical_rank]
        self.S = S[: self.numerical_rank]

    @jax.jit
    def _mahalanobis_distance(
            self,
            x: Array,
            y: Array,
    ) -> Array:
        """
        Compute Mahalanobis distance between two points x, y,
        Assume x-y-corpus_mean in the column subspace of Vt.T
        :param x: ndarray, dimension 1*n
        :param y: ndarray, dimension 1*n
        :return: The mahalanobis distance
        """
        diff = x - y
        return diff @ self.Vt.T @ jnp.diag(self.S ** (-2)) @ self.Vt @ diff.T

    @jax.jit
    def min_dist_to_corpus(
            self,
            x: ArrayLike,
            subspace_threshold: float = 1e-3,
    ) -> Array:
        # decide if x is in the subspace, mean + row_span(X - mean)
        y = x - self.corpus_mean
        rho = jnp.linalg.norm(y - y @ self.Vt.T @ self.Vt) / jnp.linalg.norm(y)

        # compute the minimal distance of x to all data points in the corpus using Mahalanobis distance
        distance_func = jax.vmap(self._mahalanobis_distance, in_axes=(None, 0))
        return jax.lax.cond(
            rho <= subspace_threshold,  # if x is in the subspace, this is true
            lambda x: jnp.min(
                distance_func(x, self.corpus)
            ),  # min distance to the corpus
            lambda x: jnp.inf,  # if x is not in the subspace, we return inf
            operand=x,
        )


class Euclidean(eqx.Module):
    corpus_mean: Array  # Mean of the corpus
    corpus: ArrayLike  # Corpus used to compute the Data

    def __init__(self, corpus: ArrayLike):
        self.corpus = corpus
        self.corpus_mean = jnp.mean(corpus, axis=0)

    @jax.jit
    def _euclidean_distance(
            self,
            x: Array,
            y: Array,
    ) -> Array:
        """
        Compute Euclidean distance between two points x, y,
        :param x: ndarray, dimension 1*n
        :param y: ndarray, dimension 1*n
        :return: The Euclidean distance
        """
        return jax.scipy.spatial.distance.euclidean(x, y)

    @jax.jit
    def min_dist_to_corpus(
            self,
            x: ArrayLike,
    ) -> Array:
        # compute the minimal distance of x to all data points in the corpus using euclidean distance
        distance_func = jax.vmap(self._euclidean_distance, in_axes=(None, 0))
        return jnp.min(distance_func(x, self.corpus))
