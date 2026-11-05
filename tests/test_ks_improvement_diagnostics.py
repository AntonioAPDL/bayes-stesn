"""Small exact-contract tests; constructed posterior states, no MCMC fitting."""
from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from bayes_stesn.data import SpatioTemporalData as Panel
from bayes_stesn.io import content_hash, seed_for
from bayes_stesn.model import FittedModel
from bayes_stesn.priors import PriorRuntime
from bayes_stesn.sampling import RegressionPosterior
from bayes_stesn.applications.ks_native import NativeFit, _resolve_bayes, _prepare_delivered_scale
from bayes_stesn.applications.ks_improvement_diagnostics import (
    conditional_mean_parameters, diagnostic_forecast, iter_diagnostic_forecasts,
    teacher_forced_readout, training_readout_residuals, spatial_design_diagnostics,
    geometry_diagnostics, synchronization_diagnostics,
)


@pytest.fixture(scope="module")
def fitted():
    angle = np.arange(6)*2*np.pi/6
    coords = np.column_stack((np.cos(angle),np.sin(angle)))
    time = np.arange(32)
    y = np.sin(.17*time[:,None]+angle)+.25*np.cos(.29*time[:,None]-2*angle)
    data = Panel(y,coords,time*.25,metadata={"ks_protocol":"interleaved","data_identity_sha256":"synthetic-only"})
    cfg,spec,sampler = _resolve_bayes({"conditioning":10,
        "features":{"modes":2,"units":[4],"squares":2},
        "sampler":{"chains":2,"draws":2,"warmup":0},"continuation_draws":0},19)
    prepared = _prepare_delivered_scale(data,cfg,spec)
    runtime = PriorRuntime(spec.prior,prepared.Phi.shape[1],prepared.Z.shape[1],
                           prepared.groups,prepared.Phi,prepared.reference_Z)
    states=[]
    for chain in range(2):
        rows=[]
        for draw in range(2):
            prior=runtime.initial_state();prior["log_precision"] += .25*(chain+draw)
            covariance=spec.covariance.initial_parameters(coords)
            covariance["log_ell"] = np.log(.5)+.1*chain
            covariance["log_q"] = -.5+.2*draw
            rows.append({"prior_state":prior,"covariance_parameters":covariance})
        states.append(rows)
    posterior = RegressionPosterior(prepared.Y,prepared.Phi,prepared.fit_Z,coords,
        prepared.groups,spec.prior,spec.covariance,sampler,states,np.zeros((2,2)),
        np.zeros(2),{},prepared.reference_Z)
    pid=content_hash({"prepared":prepared.identity,"states":states})
    core=FittedModel(prepared,posterior,sampler,{},pid)
    return NativeFit("bayes_stesn",core,cfg,19,"synthetic-native-fit", "synthetic-input",{},
                     {"qualified":False})


@pytest.fixture(scope="module")
def point(fitted):
    return conditional_mean_parameters(fitted)


def history(fitted, length=12):
    data=fitted.training
    return Panel(data.y[-length:],data.coordinates,data.times[-length:],data.site_ids,
                 metadata=dict(data.metadata))


def targets():
    angle=np.array([.3,1.1])
    return np.column_stack((np.cos(angle),np.sin(angle))),("new0","new1")


def test_conditional_readout_matches_direct_dense_gaussian_normal_equations(fitted,point):
    p=fitted.core.prepared;parameter=point.parameter
    R=p.spec.covariance.covariance(p.data.coordinates,parameter.covariance_parameters)
    precision=fitted.core.posterior._runtime.precision(parameter.prior_state,R)
    gram_s=p.Phi.T@np.linalg.solve(R,p.Phi)
    gram_t=p.fit_Z.T@p.fit_Z
    rhs=p.Phi.T@np.linalg.solve(R,p.Y)@p.fit_Z
    matrix=np.kron(gram_t,gram_s)+float(precision.values)*np.eye(rhs.size)
    expected=np.linalg.solve(matrix,rhs.reshape(-1,order="F")).reshape(rhs.shape,order="F")
    np.testing.assert_allclose(parameter.C,expected,rtol=5e-10,atol=5e-10)
    assert parameter.sigma2 == point.metadata["conditional_IG_scale"]/(point.metadata["conditional_IG_shape"]-1)
    assert point.metadata["not_full_posterior_predictive_mean"] is True
    assert point.metadata["original_fit_qualified"] is False


def test_full_mode_is_exact_original_native_predictive(fitted):
    coords,ids=targets();h=history(fitted)
    before=content_hash(fitted.to_state())
    actual=diagnostic_forecast(fitted,h,4,coords,ids,mode="full_predictive",n_paths=6,seed=73)
    expected=fitted.forecast(h,4,coords,ids,n_paths=6,seed=73)
    np.testing.assert_array_equal(actual.samples,expected.samples)
    np.testing.assert_array_equal(actual.mean,expected.mean)
    assert not actual.metadata["probabilistic_ranking_eligible"]  # fixture is deliberately not MCMC-qualified
    assert content_hash(fitted.to_state())==before


@pytest.mark.parametrize("mode",["full_predictive","parameter_only","fixed_parameter_innovations"])
def test_global_path_batching_and_resume_preserve_complete_paths(fitted,point,mode):
    coords,ids=targets();kwargs={"mode":mode,"seed":81}
    if mode=="fixed_parameter_innovations":kwargs["point"]=point
    whole=diagnostic_forecast(fitted,history(fitted),5,coords,ids,n_paths=7,**kwargs)
    pieces=list(iter_diagnostic_forecasts(fitted,history(fitted),5,coords,ids,n_paths=7,batch_size=3,**kwargs))
    np.testing.assert_array_equal(whole.samples,np.concatenate([p.samples for p in pieces]))
    resumed=diagnostic_forecast(fitted,history(fitted),5,coords,ids,n_paths=3,path_start=4,**kwargs)
    np.testing.assert_array_equal(resumed.samples,whole.samples[4:])


def test_parameter_only_shares_original_draws_and_holds_them_static(fitted):
    coords,ids=targets();h=history(fitted)
    with patch.object(fitted.core.posterior,"draw_parameters",wraps=fitted.core.posterior.draw_parameters) as draw:
        actual=diagnostic_forecast(fitted,h,5,coords,ids,mode="parameter_only",n_paths=4,seed=33)
        assert draw.call_count==4
    full=diagnostic_forecast(fitted,h,1,coords,ids,mode="full_predictive",n_paths=4,seed=33)
    assert actual.metadata["parameter_draw_ids"]==full.metadata["parameter_draw_ids"]
    p=fitted.core.prepared
    states,_=p.features.history(h.y)
    _,z=p.features.update(h.y[-1],states[-1])
    rng=np.random.default_rng(seed_for(33,"parameter",0))
    c,d=int(rng.integers(2)),int(rng.integers(2))
    parameter=fitted.core.posterior.draw_parameters(c,d,rng)
    np.testing.assert_allclose(actual.samples[0,0],p.basis.transform(coords)@parameter.C@z,rtol=1e-13,atol=1e-13)
    assert actual.metadata["diagnostic_only"]
    assert not actual.metadata["probabilistic_ranking_eligible"]


def test_fixed_parameter_innovations_reuse_exact_original_simulator(fitted,point):
    coords,ids=targets();h=history(fitted)
    actual=diagnostic_forecast(fitted,h,4,coords,ids,mode="fixed_parameter_innovations",
                               point=point,n_paths=5,seed=41)
    with patch.object(fitted.core.posterior,"draw_parameters",return_value=point.parameter):
        expected=fitted.forecast(h,4,coords,ids,n_paths=5,seed=41)
    np.testing.assert_array_equal(actual.samples,expected.samples)


def test_deterministic_and_teacher_forced_first_lead_agree_but_only_past_inputs_are_used(fitted,point):
    h=history(fitted);coords,ids=targets()
    future=Panel(np.tile(h.y[-1],(4,1)),h.coordinates,h.times[-1]+.25*np.arange(1,5),
                 h.site_ids,metadata=dict(h.metadata))
    deterministic=diagnostic_forecast(fitted,h,4,coords,ids,mode="deterministic_conditional_mean",n_paths=1,point=point)
    forced=teacher_forced_readout(fitted,h,future,coords,ids,point=point)
    np.testing.assert_allclose(deterministic.mean[0],forced.mean[0],rtol=1e-13,atol=1e-13)
    changed=future.y.copy();changed[1:,0]+=10
    after=teacher_forced_readout(fitted,h,Panel(changed,future.coordinates,future.times,future.site_ids,
        metadata=dict(future.metadata)),coords,ids,point=point)
    np.testing.assert_array_equal(after.mean[:2],forced.mean[:2])
    assert not np.allclose(after.mean[2:],forced.mean[2:])
    assert deterministic.samples is None
    assert "NOT_predictive_mean" in deterministic.metadata["mean_policy"]
    with pytest.raises(ValueError,match="n_paths=1"):
        diagnostic_forecast(fitted,h,2,mode="deterministic_conditional_mean",n_paths=2)


def test_mutated_point_wrong_trajectory_and_oversized_output_are_refused(fitted,point):
    changed=copy.deepcopy(point);changed.parameter.sigma2*=2
    with pytest.raises(ValueError,match="mutated"):
        diagnostic_forecast(fitted,history(fitted),2,mode="deterministic_conditional_mean",n_paths=1,point=changed)
    h=history(fitted);wrong=Panel(h.y,h.coordinates,h.times,h.site_ids,metadata={**h.metadata,"data_identity_sha256":"other"})
    with pytest.raises(ValueError,match="identity"):
        diagnostic_forecast(fitted,wrong,2,mode="parameter_only",n_paths=1)
    with pytest.raises(MemoryError):
        diagnostic_forecast(fitted,h,100,n_paths=100,max_output_bytes=32)


def test_geometry_known_condition_and_decoder_regularization():
    result=spatial_design_diagnostics(np.diag([1.,1e-8]),np.array([[0.,1.]]),ridge_values=(0.,.1))
    assert result["condition_number"]==1e8
    assert result["weak_directions_including_unidentified"]==1
    np.testing.assert_allclose(result["decoder_diagnostics"][0]["operator_norm"],1e8)
    np.testing.assert_allclose(result["decoder_diagnostics"][1]["operator_norm"],1e-8/(1e-16+.2))
    null=spatial_design_diagnostics(np.array([[1.,0.,0.],[0.,1.,0.]]),np.array([[0.,0.,1.]]))
    assert null["condition_number"] is None
    assert null["weak_directions_including_unidentified"]==1
    assert null["target_unresolved_direction_row_norms"]==[1.]
    with pytest.raises(MemoryError):
        spatial_design_diagnostics(np.eye(3),np.eye(3),max_matrix_bytes=1)


def test_unfitted_feature_shell_supports_geometry_and_existing_synchronization(fitted):
    shell=copy.copy(fitted);shell.core=SimpleNamespace(prepared=fitted.core.prepared)
    result=geometry_diagnostics(shell,targets()[0])
    assert result["has_fitted_posterior"] is False
    assert result["uses_response_values"] is False
    synced=synchronization_diagnostics(shell,fitted.training,lengths=(12,32,64))
    assert synced["rows"][1]["passed"] and synced["rows"][1]["state_max_absolute_difference"]==0
    assert not synced["rows"][2]["available"]
    residual=training_readout_residuals(fitted)
    assert residual["rows"]==22 and "in_sample" in residual["regime"]


def test_unsupported_innovation_hierarchy_is_not_silently_approximated(fitted):
    p=fitted.core.prepared
    changed=copy.copy(fitted)
    changed.core=replace(fitted.core,prepared=replace(p,spec=replace(p.spec,innovations="full_residual_ar")))
    with pytest.raises(ValueError,match="accepted iid"):
        diagnostic_forecast(changed,history(fitted),2,n_paths=1)

