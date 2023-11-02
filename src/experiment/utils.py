import itertools
from typing import Iterator, TypeVar, Iterable

import numpy as np
from scipy.stats import norm

T = TypeVar("T")


def batched(iterable: Iterable[T], batch_size: int) -> Iterator[Iterator[T]]:
    """
    Lazily collect data from iterable into non-overlapping batch iterators
    Example: ([1, 2, 3, 4, 5], 2) -> ([1, 2], [3, 4], [5])

    Parameters:
    - iterable (Iterable[T]): The source iterable.
    - batch_size (int): The size of each batch.

    Returns:
    - Iterator[Iterator[T]]: An iterator over the batches of elements.
    """
    iterator = iter(iterable)
    for first in iterator:
        rest = itertools.islice(iterator, batch_size - 1)
        yield itertools.chain([first], rest)


def random_sample(
    iterator: Iterator[T],
    iterator_size: int,
    rng: np.random.Generator,
    sample_size: int
) -> Iterator[T]:
    """
    Return an iterator of `sample_size` pseudo-random samples elements from `iterator`
    using a permutation-based approach to ensure samples from a smaller sample size
    are included in larger sample sizes for the same seed.

    Parameters:
    - iterator (Iterator[T]): The source iterator.
    - iterator_size (int): Total number of items in the iterator.
    - rng (np.random.Generator): A seeded random number generator for reproducibility.
    - sample_size (int): The number of items to sample.

    Returns:
    - Iterator[T]: An iterator over the selected sample of elements.
    """
    # Generate a permutation of indices based on the seed
    permuted_indices = rng.permutation(iterator_size)
    # Select the first `sample_size` indices from the permuted list
    selected_indices = set(permuted_indices[:sample_size])
    # Yield elements whose indices are in selected_indices
    return (element for i, element in enumerate(iterator) if i in selected_indices)


def compute_confidence_interval(successes: int, total: int, confidence_level: float = 0.95) -> tuple[float, float]:
    """
    Compute the confidence interval for a proportion (accuracy).

    Parameters:
    - successes: The number of successful trials (e.g., correct classifications).
    - total: The total number of trials.
    - confidence_level: The desired confidence level for the interval.

    Returns:
    - A tuple containing the lower and upper bounds of the confidence interval.
    """
    # Calculate the sample proportion (accuracy)
    p_hat = successes / total

    # todo note: only works for bernoulli
    standard_error = (p_hat * (1 - p_hat) / total) ** 0.5

    # Lookup the z-score for the given confidence level
    z_score = norm.ppf((1 + confidence_level) / 2)

    margin_of_error = z_score * standard_error

    # Calculate the confidence interval
    ci_lower = p_hat - margin_of_error
    ci_upper = p_hat + margin_of_error

    return ci_lower, ci_upper


def compute_initial_sample_size(
    estimated_proportion: float,
    margin_of_error: float,
    confidence_level: float = 0.95
) -> int:
    """
    Compute the initial sample size needed for a given estimated proportion,
    margin of error, and confidence level.

    Parameters:
    - estimated_proportion: The estimated proportion of success (e.g., accuracy).
    - margin_of_error: The desired margin of error (half-width of the confidence interval).
    - confidence_level: The desired confidence level for the estimate.

    Returns:
    - The calculated sample size (rounded up to the nearest whole number).
    """
    # Lookup the Z-score for the given confidence level
    z_score = norm.ppf((1 + confidence_level) / 2)

    # Compute the sample size
    sample_size = (z_score ** 2 * estimated_proportion * (1 - estimated_proportion)) / (margin_of_error ** 2)

    # Round up to the nearest integer
    return int(sample_size + 0.5)
