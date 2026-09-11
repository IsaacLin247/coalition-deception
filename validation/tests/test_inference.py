import itertools

import numpy as np
import pytest

from validation.inference import exact_sign_flip, holm, summarize_contrast


@pytest.mark.parametrize("values", [
    [0, 0, 0], [1, 1, 1, 1, 1], [-5, 3, 1, 0],
    [2, -2, 3, -3], [-14, 7, 21, -7], [1, 2, 4, 8, 16, -32],
])
def test_exact_dp_agrees_with_independent_enumeration(values):
    observed = abs(sum(values))
    extreme = sum(abs(sum(s * v for s, v in zip(signs, values))) >= observed
                  for signs in itertools.product((-1, 1), repeat=len(values)))
    result = exact_sign_flip(values)
    assert int(result["extreme_assignments"]) == extreme
    assert int(result["total_assignments"]) == 2 ** len(values)
    assert result["p_exact"] == extreme / 2 ** len(values)


def test_large_cohort_preserves_integer_counts_and_zero_multiplicity():
    positive = exact_sign_flip([1] * 72)
    assert int(positive["total_assignments"]) == 2 ** 72
    assert int(positive["extreme_assignments"]) == 2
    assert positive["p_exact"] == 2 ** -71
    zeros = exact_sign_flip([1] * 5 + [0] * 67)
    assert zeros["p_exact"] == .0625


def test_holm_known_family_and_permutation():
    ps = [.01, .04, .03, .002]
    assert holm(ps) == [.03, .06, .06, .008]
    assert holm(ps[::-1]) == holm(ps)[::-1]


def test_inference_rejects_partial_cohort_and_floating_counts():
    with pytest.raises(ValueError):
        exact_sign_flip([.1, .2])
    with pytest.raises(ValueError):
        summarize_contrast("test", [1, 2], 1000, [1000, 1001, 1002], [1000, 1001])


def test_bootstrap_is_reproducible_and_retains_negative_effect():
    seeds = list(range(1000, 1072))
    result = summarize_contrast("test", [-20] * 72, 1000, seeds, seeds)
    assert result == summarize_contrast("test", [-20] * 72, 1000, seeds, seeds)
    np.testing.assert_allclose(result["mean"], -.02)
    np.testing.assert_allclose(result["ci95_pointwise"], [-.02, -.02])
