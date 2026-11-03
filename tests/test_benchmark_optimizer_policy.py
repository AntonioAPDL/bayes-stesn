"""Convergence caps remain scientific failures, not finite-fit successes."""
from types import SimpleNamespace
import numpy as np
import pytest

from bayes_stesn.benchmark.optimization import bounded_lbfgsb, projected_gradient
from bayes_stesn.benchmark.classical import fit_classical
from bayes_stesn.benchmark.contracts import Panel


def test_projected_gradient_respects_bound_optima():
    assert projected_gradient([0.,1.],[2.,-3.],[[0.,1.],[0.,1.]])==0.
    assert projected_gradient([0.,1.],[-2.,3.],[[0.,1.],[0.,1.]])==3.


def test_one_improving_iteration_cap_continuation_preserves_tolerances():
    objective=lambda x: float((x[0]-.2)**2+10*(x[1]+.3)**2)
    initial=np.array([3.,-4.]);bounds=np.array([[-5.,5.],[-5.,5.]])
    first,limited=bounded_lbfgsb(objective,initial,bounds,max_iter=1,ftol=1e-12,gtol=1e-7)
    result,rescued=bounded_lbfgsb(objective,initial,bounds,max_iter=1,ftol=1e-12,gtol=1e-7,
                                continuation_max_iter=100)
    assert not limited["qualified"] and limited["continuations"]==0
    assert rescued["qualified"] and rescued["continuations"]==1
    assert rescued["phases"][0]["objective"]==limited["phases"][0]["objective"]
    assert all(p["ftol"]==1e-12 and p["gtol"]==1e-7 for p in rescued["phases"])
    np.testing.assert_allclose(result.x,[.2,-.3],atol=1e-5)


def test_abnormal_exit_is_not_automatically_restarted(monkeypatch):
    import bayes_stesn.benchmark.optimization as module
    calls=[]
    def failed(*args,**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(x=np.array([0.]),fun=0.,jac=np.array([1.]),
                               success=False,status=2,message="abnormal line search",nit=1,nfev=3)
    monkeypatch.setattr(module,"minimize",failed)
    _,info=bounded_lbfgsb(lambda x:float(np.dot(x,x)),[2.],[[0.,3.]],max_iter=1,
                         continuation_max_iter=20,ftol=1e-8)
    assert len(calls)==1 and not info["qualified"]


def test_nonfinite_penalty_cannot_qualify_even_if_optimizer_stops():
    _,info=bounded_lbfgsb(lambda x:1e100,[0.],[[-1.,1.]],max_iter=2,ftol=1e-8,
                         continuation_max_iter=10)
    assert not info["qualified"] and info["continuations"]==0


def test_lgssm_larger_cap_preserves_earlier_likelihood_sequence():
    rng=np.random.default_rng(50)
    data=Panel(rng.normal(size=(60,8)),rng.uniform(size=(8,2)))
    limited=fit_classical("basis_lgssm",data,{"rank":3,"max_iter":2,"tolerance":1e-12})
    extended=fit_classical("basis_lgssm",data,{"rank":3,"max_iter":8,"tolerance":1e-12})
    prefix=limited.diagnostics["log_likelihood"]
    np.testing.assert_array_equal(prefix,extended.diagnostics["log_likelihood"][:len(prefix)])
    assert limited.diagnostics["stop_reason"]=="iteration_cap"
    assert not limited.diagnostics["qualified"]
    assert extended.diagnostics["optimizer_budget"]["tolerance"]==1e-12


def test_gp_records_bounded_phase_costs_and_projection():
    rng=np.random.default_rng(51)
    data=Panel(rng.normal(size=(40,5)),rng.uniform(size=(5,2)))
    fit=fit_classical("separable_stgp",data,{"max_iter":1,"continuation_max_iter":2})
    phases=fit.diagnostics["optimizer_details"]["phases"]
    assert 1<=len(phases)<=2
    assert fit.diagnostics["iterations"]==sum(p["iterations"] for p in phases)
    assert fit.diagnostics["optimizer_projected_gradient_inf"] is not None
    assert fit.diagnostics["qualified"]==fit.diagnostics["optimizer_details"]["qualified"]
    with pytest.raises(ValueError,match="nonnegative"):
        fit_classical("separable_stgp",data,{"continuation_max_iter":-1})
