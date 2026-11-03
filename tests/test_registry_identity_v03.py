"""The named ridge benchmark cannot silently become a different model."""
from unittest import mock

import numpy as np
import pytest

from bayes_stesn.basis import SpatialMeanSpec
from bayes_stesn.benchmark.contracts import Panel
from bayes_stesn.benchmark.esn import fit_esn
from bayes_stesn.benchmark.registry import fit_model, resolve_model_config, validate_model_options
from bayes_stesn.data import SplitPlan
from bayes_stesn.features import FeatureSpec
from bayes_stesn.model import ModelSpec, fit, prepare
from bayes_stesn.priors import PriorSpec
from bayes_stesn.sampling import SamplerConfig


@pytest.fixture
def panel():
    rng = np.random.default_rng(1307)
    return Panel(rng.normal(size=(50, 4)), rng.uniform(size=(4, 2)))


@pytest.mark.parametrize("options", [
    {"prior":{"kind":"group"}},
    {"prior":{"kind":"rhs", "unit":"group"}},
    {"innovations":"full_residual_ar"},
    {"innovations":"latent_ar"},
    {"innovations":"field_t"},
])
def test_nonbaseline_settings_fail_before_named_benchmark_fitting(panel, tmp_path, options):
    with pytest.raises(ValueError, match="scalar-ridge iid Gaussian"):
        validate_model_options("bayes_esn_ridge", options)
    with pytest.raises(ValueError, match="core prepare/fit"):
        resolve_model_config("bayes_esn_ridge", options, fitting_sites=4, fitting_rows=50)
    with mock.patch("bayes_stesn.benchmark.esn.prepare") as preprocessing:
        for invoke in (lambda:fit_esn(panel, options, checkpoint_path=tmp_path/"chain.npz"),
                       lambda:fit_model("bayes_esn_ridge", panel, options,
                                        checkpoint_path=tmp_path/"chain.npz")):
            with pytest.raises(ValueError, match="core prepare/fit"):
                invoke()
        preprocessing.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_baseline_guard_does_not_remove_declared_covariance_options():
    for covariance in ({"kind":"matern"}, {"kind":"vecchia"},
                       {"scale_mode":"separate", "range_prior":"pc"}):
        validate_model_options("bayes_esn_ridge", {
            "prior":{"kind":"ridge", "sd_log_precision":.8},
            "innovations":"iid", "covariance":covariance})


@pytest.mark.parametrize("extension", ["grouped", "full_residual_ar"])
def test_core_api_keeps_extensions_available(panel, extension):
    spec = ModelSpec(features=FeatureSpec(modes=2, units=(4,), squares=2, lags=(1,)),
        spatial_mean=SpatialMeanSpec(side=2),
        prior=PriorSpec.grouped() if extension == "grouped" else PriorSpec.ridge(),
        innovations="full_residual_ar" if extension == "full_residual_ar" else "iid")
    prepared = prepare(panel, SplitPlan(12, len(panel.y)), spec)
    fitted = fit(prepared, SamplerConfig(chains=1, warmup=2, draws=3, initialize="prior"))
    assert fitted.posterior.n_draws == 3
    assert fitted.prepared.spec == spec
    if extension == "grouped":
        assert any(name.startswith("prior.log_precision[") for name in fitted.posterior.parameter_names)
    else:
        assert "temporal.rho" in fitted.posterior.parameter_names
