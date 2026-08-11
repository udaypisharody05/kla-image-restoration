"""Fast, dataset-free tests for the degradation-analysis utilities (src/degradation.py).

Analysis-only code: nothing here trains or evaluates a model.
"""

import numpy as np
import pytest

from src.degradation import (
    BinnedAccumulator,
    MomentAccumulator,
    apply_gain_bias,
    autocorrelation_at_offsets,
    available_downsamplers,
    connected_components,
    content_hash,
    downsample,
    fit_gain_bias,
    fit_noise_variance_model,
    gaussian_blur,
    gradient_magnitude,
    local_variance,
    match_metrics,
    perceptual_signature,
    power_spectrum,
    radial_profile,
    residual_moments,
)


# --- Downsampling models ---


def test_every_downsampler_halves_both_spatial_dimensions() -> None:
    gt = np.random.default_rng(0).uniform(0, 1, size=(64, 64)).astype(np.float32)
    for method in available_downsamplers():
        assert downsample(gt, method).shape == (32, 32), method


def test_area_downsampler_is_the_2x2_block_mean() -> None:
    gt = np.arange(16, dtype=np.float32).reshape(4, 4)
    expected = gt.reshape(2, 2, 2, 2).mean(axis=(1, 3))
    assert np.allclose(downsample(gt, "area"), expected)


def test_subsample_even_takes_the_even_grid() -> None:
    gt = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert np.array_equal(downsample(gt, "subsample_even"), gt[0::2, 0::2])


def test_downsampling_a_constant_image_preserves_the_constant() -> None:
    gt = np.full((32, 32), 0.37, dtype=np.float32)
    for method in available_downsamplers():
        assert np.allclose(downsample(gt, method), 0.37, atol=1e-5), method


def test_downsample_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown downsampling method"):
        downsample(np.zeros((8, 8), dtype=np.float32), "not_a_method")


def test_downsample_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="2D"):
        downsample(np.zeros((4, 8, 8), dtype=np.float32), "bicubic")


def test_downsample_rejects_odd_dimensions() -> None:
    with pytest.raises(ValueError, match="even"):
        downsample(np.zeros((7, 8), dtype=np.float32), "bicubic")


# --- Gaussian blur ---


def test_zero_sigma_blur_is_identity() -> None:
    image = np.random.default_rng(1).uniform(0, 1, size=(16, 16))
    assert np.array_equal(gaussian_blur(image, 0.0), image)


def test_blur_preserves_shape_and_mean_of_a_constant_image() -> None:
    image = np.full((24, 24), 0.5)
    blurred = gaussian_blur(image, 1.0)
    assert blurred.shape == image.shape
    assert np.allclose(blurred, 0.5, atol=1e-9)


def test_blur_reduces_high_frequency_variance() -> None:
    rng = np.random.default_rng(2)
    noisy = rng.normal(0.5, 0.2, size=(48, 48))
    assert gaussian_blur(noisy, 1.5).var() < noisy.var()


# --- Match metrics / affine fit ---


def test_match_metrics_are_zero_for_identical_arrays() -> None:
    image = np.random.default_rng(3).uniform(0, 1, size=(16, 16))
    metrics = match_metrics(image, image)
    assert metrics["mae"] == 0.0
    assert metrics["mse"] == 0.0
    assert metrics["bias"] == 0.0
    assert metrics["psnr"] == float("inf")
    assert metrics["correlation"] == pytest.approx(1.0)


def test_match_metrics_bias_is_signed_mean_difference() -> None:
    estimate = np.zeros((8, 8))
    observed = np.full((8, 8), 0.25)
    assert match_metrics(observed, estimate)["bias"] == pytest.approx(0.25)


def test_match_metrics_reports_finite_correlation_for_constant_inputs() -> None:
    """Correlation is mathematically undefined here; it must degrade to a
    JSON-serializable 0.0 rather than NaN."""
    metrics = match_metrics(np.full((8, 8), 0.25), np.zeros((8, 8)))
    assert metrics["correlation"] == 0.0


def test_match_metrics_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="Shape mismatch"):
        match_metrics(np.zeros((4, 4)), np.zeros((4, 5)))


def test_fit_gain_bias_recovers_an_exact_affine_relationship() -> None:
    rng = np.random.default_rng(4)
    estimate = rng.uniform(0, 1, size=(32, 32))
    observed = 1.35 * estimate - 0.07
    gain, bias = fit_gain_bias(observed, estimate)
    assert gain == pytest.approx(1.35, abs=1e-9)
    assert bias == pytest.approx(-0.07, abs=1e-9)


def test_fit_gain_bias_is_identity_for_identical_inputs() -> None:
    estimate = np.random.default_rng(5).uniform(0, 1, size=(16, 16))
    gain, bias = fit_gain_bias(estimate, estimate)
    assert gain == pytest.approx(1.0)
    assert bias == pytest.approx(0.0, abs=1e-12)


def test_fit_gain_bias_handles_a_constant_estimate_without_dividing_by_zero() -> None:
    gain, bias = fit_gain_bias(np.full((8, 8), 0.4), np.full((8, 8), 0.1))
    assert gain == 1.0
    assert bias == pytest.approx(0.3)


def test_apply_gain_bias_matches_the_fitted_relationship() -> None:
    estimate = np.random.default_rng(6).uniform(0, 1, size=(16, 16))
    observed = 0.8 * estimate + 0.05
    gain, bias = fit_gain_bias(observed, estimate)
    assert np.allclose(apply_gain_bias(estimate, gain, bias), observed, atol=1e-9)


# --- Residual statistics ---


def test_residual_moments_match_numpy_for_gaussian_noise() -> None:
    values = np.random.default_rng(7).normal(0.0, 0.1, size=(256, 256))
    moments = residual_moments(values)
    assert moments["mean"] == pytest.approx(values.mean())
    assert moments["std"] == pytest.approx(values.std())
    assert moments["min"] == values.min()
    assert moments["max"] == values.max()
    # Gaussian noise: skewness ~0, excess kurtosis ~0.
    assert abs(moments["skewness"]) < 0.1
    assert abs(moments["excess_kurtosis"]) < 0.1


def test_residual_moments_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        residual_moments(np.array([]))


def test_white_noise_has_near_zero_autocorrelation() -> None:
    noise = np.random.default_rng(8).normal(size=(256, 256))
    values = autocorrelation_at_offsets(noise, [(0, 1), (1, 0), (2, 2)])
    assert all(abs(v) < 0.05 for v in values.values()), values


def test_horizontally_constant_field_is_perfectly_correlated_along_rows() -> None:
    column = np.random.default_rng(9).normal(size=(64, 1))
    field = np.repeat(column, 64, axis=1)  # identical along each row
    values = autocorrelation_at_offsets(field, [(0, 1), (0, 8)])
    assert values["0_1"] == pytest.approx(1.0, abs=1e-9)
    assert values["0_8"] == pytest.approx(1.0, abs=1e-9)


def test_autocorrelation_rejects_negative_and_oversized_offsets() -> None:
    field = np.zeros((8, 8))
    with pytest.raises(ValueError, match="non-negative"):
        autocorrelation_at_offsets(field, [(-1, 0)])
    with pytest.raises(ValueError, match="exceeds"):
        autocorrelation_at_offsets(field, [(9, 0)])


# --- Spectrum ---


def test_power_spectrum_of_a_single_sinusoid_peaks_off_centre() -> None:
    size = 32
    x = np.arange(size)
    field = np.sin(2 * np.pi * 4 * x / size)[None, :] * np.ones((size, 1))
    spectrum = power_spectrum(field)
    peak_y, peak_x = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    assert peak_y == size // 2  # no vertical variation
    assert peak_x != size // 2  # energy sits at a non-zero horizontal frequency


def test_radial_profile_length_and_finiteness() -> None:
    spectrum = power_spectrum(np.random.default_rng(10).normal(size=(64, 64)))
    profile = radial_profile(spectrum)
    assert profile.ndim == 1
    assert len(profile) >= 32
    assert np.isfinite(profile).all()


# --- Local structure ---


def test_gradient_magnitude_is_zero_on_a_constant_image() -> None:
    assert np.allclose(gradient_magnitude(np.full((16, 16), 0.3)), 0.0)


def test_gradient_magnitude_detects_a_step_edge() -> None:
    image = np.zeros((16, 16))
    image[:, 8:] = 1.0
    magnitude = gradient_magnitude(image)
    assert magnitude[:, 7:9].max() > 0.4
    assert magnitude[:, :6].max() == pytest.approx(0.0)


def test_local_variance_is_zero_on_a_constant_image() -> None:
    assert np.allclose(local_variance(np.full((16, 16), 0.7), window=3), 0.0, atol=1e-12)


def test_local_variance_is_nonnegative_and_shape_preserving() -> None:
    image = np.random.default_rng(11).uniform(0, 1, size=(24, 24))
    result = local_variance(image, window=3)
    assert result.shape == image.shape
    assert (result >= 0).all()


def test_local_variance_rejects_even_windows() -> None:
    with pytest.raises(ValueError, match="odd"):
        local_variance(np.zeros((8, 8)), window=4)


# --- Hashing / duplicate detection ---


def test_content_hash_is_stable_and_distinguishes_arrays() -> None:
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    b = a.copy()
    c = a + 1
    assert content_hash(a) == content_hash(b)
    assert content_hash(a) != content_hash(c)


def test_perceptual_signature_survives_added_noise() -> None:
    """Two noise realizations of one scene must collide -- that is exactly the
    near-duplicate case the repeated-scene analysis is looking for."""
    rng = np.random.default_rng(12)
    scene = np.tile(np.linspace(0, 1, 32, dtype=np.float32), (32, 1))
    noisy = scene + rng.normal(0, 0.01, size=scene.shape).astype(np.float32)
    assert perceptual_signature(scene) == perceptual_signature(noisy)


def test_perceptual_signature_differs_for_different_scenes() -> None:
    horizontal = np.tile(np.linspace(0, 1, 32, dtype=np.float32), (32, 1))
    assert perceptual_signature(horizontal) != perceptual_signature(horizontal.T)


def test_connected_components_merges_transitive_duplicates() -> None:
    """A matches B and B matches C -> all three are one scene group."""
    groups = connected_components(["a", "b", "c", "d"], [("a", "b"), ("b", "c")])
    assert groups == [["a", "b", "c"], ["d"]]


def test_connected_components_without_edges_yields_singletons() -> None:
    assert connected_components(["x", "y"], []) == [["x"], ["y"]]


def test_connected_components_is_deterministic_regardless_of_edge_order() -> None:
    nodes = ["a", "b", "c", "d", "e"]
    forward = connected_components(nodes, [("a", "b"), ("d", "e"), ("b", "c")])
    reversed_order = connected_components(nodes, [("b", "c"), ("d", "e"), ("a", "b")])
    assert forward == reversed_order == [["a", "b", "c"], ["d", "e"]]


def test_connected_components_rejects_edges_referencing_unknown_nodes() -> None:
    with pytest.raises(KeyError, match="unknown node"):
        connected_components(["a"], [("a", "ghost")])


# --- Streaming accumulators ---


def test_moment_accumulator_matches_direct_computation() -> None:
    rng = np.random.default_rng(13)
    chunks = [rng.normal(0.01, 0.08, size=(64, 64)) for _ in range(5)]
    accumulator = MomentAccumulator()
    for chunk in chunks:
        accumulator.update(chunk)
    summary = accumulator.summary()
    combined = np.concatenate([chunk.ravel() for chunk in chunks])
    assert summary["count"] == combined.size
    assert summary["mean"] == pytest.approx(combined.mean(), abs=1e-9)
    assert summary["std"] == pytest.approx(combined.std(), abs=1e-9)
    assert summary["min"] == combined.min()
    assert summary["max"] == combined.max()


def test_moment_accumulator_percentiles_are_close_to_exact() -> None:
    values = np.random.default_rng(14).normal(0.0, 0.1, size=200_000)
    accumulator = MomentAccumulator()
    accumulator.update(values)
    summary = accumulator.summary()
    # Histogram-based, so accurate to about one bin width (0.0005).
    assert summary["percentiles"]["p50"] == pytest.approx(np.percentile(values, 50), abs=2e-3)
    assert summary["percentiles"]["p95"] == pytest.approx(np.percentile(values, 95), abs=2e-3)


def test_moment_accumulator_rejects_summary_before_any_update() -> None:
    with pytest.raises(ValueError, match="No values"):
        MomentAccumulator().summary()


def test_binned_accumulator_computes_per_bin_mean_and_std() -> None:
    edges = np.array([0.0, 0.5, 1.0])
    accumulator = BinnedAccumulator(edges)
    accumulator.update(np.array([0.1, 0.2, 0.7, 0.8]), np.array([1.0, 3.0, 10.0, 20.0]))
    rows = accumulator.summary()
    assert rows[0]["count"] == 2
    assert rows[0]["mean"] == pytest.approx(2.0)
    assert rows[0]["std"] == pytest.approx(1.0)
    assert rows[1]["mean"] == pytest.approx(15.0)


def test_binned_accumulator_min_count_filters_sparse_bins() -> None:
    accumulator = BinnedAccumulator(np.array([0.0, 0.5, 1.0]))
    accumulator.update(np.array([0.1, 0.2, 0.3, 0.9]), np.array([1.0, 1.0, 1.0, 5.0]))
    assert len(accumulator.summary(min_count=2)) == 1


def test_binned_accumulator_rejects_mismatched_inputs() -> None:
    accumulator = BinnedAccumulator(np.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="equal size"):
        accumulator.update(np.array([0.1, 0.2]), np.array([1.0]))


# --- Noise variance model ---


def test_variance_model_recovers_known_quadratic_coefficients() -> None:
    intensity = np.linspace(0.0, 1.0, 25)
    variance = 0.001 + 0.02 * intensity + 0.005 * intensity**2
    fit = fit_noise_variance_model(intensity, variance)
    assert fit["constant"] == pytest.approx(0.001, abs=1e-9)
    assert fit["linear"] == pytest.approx(0.02, abs=1e-9)
    assert fit["quadratic"] == pytest.approx(0.005, abs=1e-9)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-9)


def test_variance_model_identifies_a_constant_homoscedastic_profile() -> None:
    intensity = np.linspace(0.0, 1.0, 20)
    fit = fit_noise_variance_model(intensity, np.full_like(intensity, 0.004))
    assert fit["constant"] == pytest.approx(0.004, abs=1e-9)
    assert abs(fit["linear"]) < 1e-9
    assert abs(fit["quadratic"]) < 1e-9


def test_variance_model_requires_enough_bins() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        fit_noise_variance_model([0.1, 0.2], [0.01, 0.02])
