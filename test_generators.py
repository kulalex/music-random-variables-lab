"""Automated checks for the empirical generators (deterministic seeds)."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from generators import conditional_generator, independent_generator, empirical_quantile


@pytest.fixture(scope="module")
def tracks():
    df = pd.read_csv("data/spotify_tracks_sample.csv")
    assert df.track_id.is_unique
    return df


def test_empirical_inverse_cdf():
    observed = np.array([0.2, 0.4, 0.9])
    u = np.array([0, 0.1, 1/3, 2/3, 0.999999])
    np.testing.assert_array_equal(empirical_quantile(observed, u),
                                  [0.2, 0.2, 0.4, 0.9, 0.9])


def test_invalid_uniforms():
    with pytest.raises(ValueError):
        empirical_quantile([0.1], [-0.2])
    with pytest.raises(ValueError):
        empirical_quantile([0.1], [1.0])


@pytest.mark.parametrize("kind", [conditional_generator, independent_generator])
def test_repeatability_and_range(tracks, kind):
    a = kind(tracks, n=4000, seed=123)
    b = kind(tracks, n=4000, seed=123)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 4000
    assert a.notna().all().all()
    assert a.apply(lambda c: c.between(0,1).all()).all()


def test_invalid_inputs(tracks):
    with pytest.raises(ValueError):
        conditional_generator(tracks, n=0)
    with pytest.raises(ValueError):
        conditional_generator(tracks, n=400, bins=1)
    with pytest.raises(ValueError):
        independent_generator(tracks.assign(valence=np.nan), n=400)


def test_generated_marginal_means(tracks):
    generated = conditional_generator(tracks, n=12000, seed=71)
    for name in ("danceability", "valence"):
        assert abs(generated[name].mean() - tracks[name].mean()) < .025


def test_conditional_generator_preserves_association(tracks):
    generated = conditional_generator(tracks, n=12000, seed=71)
    independent = independent_generator(tracks, n=12000, seed=71)
    real_r = spearmanr(tracks.danceability, tracks.valence).statistic
    gen_r = spearmanr(generated.danceability, generated.valence).statistic
    ind_r = spearmanr(independent.danceability, independent.valence).statistic
    assert real_r > .4
    assert abs(gen_r - real_r) < .08
    assert abs(ind_r) < .05
