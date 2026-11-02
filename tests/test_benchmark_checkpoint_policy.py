"""Checkpoint cadence changes storage work, never the sampled numerical law."""
from dataclasses import replace
from pathlib import Path
import tempfile
from unittest import mock

import numpy as np

from bayes_stesn.assessment import diagnose
from bayes_stesn.benchmark.contracts import Panel
from bayes_stesn.benchmark.esn import fit_esn
from bayes_stesn.io import content_hash,load_state,save_state,seed_for
from bayes_stesn.model import fit
from bayes_stesn.sampling import SamplerConfig


def test_benchmark_endpoint_default_preserves_override_and_exact_continuation():
    rng=np.random.default_rng(991)
    coordinates=rng.uniform(size=(6,2))
    data=Panel(rng.normal(size=(90,6)),coordinates)
    seed=541
    config={"conditioning":20,
            "features":{"modes":2,"units":[6],"squares":3,"lags":[1],"input_scales":[.5]},
            "spatial_mean":{"side":2,"width":.3},
            "sampler":{"chains":2,"warmup":20,"draws":30,"initialize":"prior"},
            "diagnostics":{"functional_draws":16,"min_chains":4},
            "continuation_draws":15}
    counts={"endpoint":0,"periodic":0}
    def tracked_save(path,state,**kwargs):
        counts[Path(path).stem]+=1
        return save_state(path,state,**kwargs)
    with tempfile.TemporaryDirectory() as temporary:
        endpoint_path=Path(temporary)/"endpoint.npz"
        periodic_path=Path(temporary)/"periodic.npz"
        with mock.patch("bayes_stesn.model.save_state",side_effect=tracked_save):
            endpoint=fit_esn(data,config,seed,checkpoint_path=endpoint_path)
            periodic=fit_esn(data,{**config,"sampler":{**config["sampler"],"checkpoint_every":10}},
                             seed,checkpoint_path=periodic_path)
        # Both passes save every chain endpoint; only the explicit override
        # performs periodic snapshots during a chain.
        assert counts=={"endpoint":4,"periodic":14}
        assert SamplerConfig().checkpoint_every==100
        assert endpoint.fitted.sampler.checkpoint_every==0
        assert periodic.fitted.sampler.checkpoint_every==10
        assert endpoint.diagnostics["checkpoint_cadence"]=={
            "every_iterations":0,"chain_endpoints_always_saved":True,"source":"benchmark_default"}
        assert periodic.diagnostics["checkpoint_cadence"]=={
            "every_iterations":10,"chain_endpoints_always_saved":True,"source":"caller"}
        assert endpoint.diagnostics["continuations"]==periodic.diagnostics["continuations"]==1
        assert not endpoint.diagnostics["qualified"]
        assert endpoint.diagnostics["gates"]==periodic.diagnostics["gates"]
        a,b=endpoint.fitted.posterior,periodic.fitted.posterior
        np.testing.assert_array_equal(a.chains,b.chains)
        np.testing.assert_array_equal(a.log_density,b.log_density)
        np.testing.assert_array_equal(a.acceptance,b.acceptance)
        assert a.n_draws==45
        for chain,draw in ((0,0),(1,29),(1,44)):
            pa=a.draw_parameters(chain,draw,np.random.default_rng(83))
            pb=b.draw_parameters(chain,draw,np.random.default_rng(83))
            assert pa.sigma2==pb.sigma2
            np.testing.assert_array_equal(pa.C,pb.C)
        targets=np.vstack([coordinates[:2],[[.25,.75]]])
        fa=endpoint.forecast(4,targets,n_paths=5,seed=18)
        fb=periodic.forecast(4,targets,n_paths=5,seed=18)
        np.testing.assert_array_equal(fa.mean,fb.mean)
        np.testing.assert_array_equal(fa.samples,fb.samples)
        ca=load_state(endpoint_path,kind="mcmc_checkpoint")
        cb=load_state(periodic_path,kind="mcmc_checkpoint")
        assert content_hash(ca["chains"])==content_hash(cb["chains"])
        # Pausing at30 draws and resuming to45 is also exactly the uninterrupted
        # kernel, independently of either file-writing cadence.
        uninterrupted=fit(endpoint.fitted.prepared,replace(endpoint.fitted.sampler,draws=45))
        np.testing.assert_array_equal(a.chains,uninterrupted.posterior.chains)
        np.testing.assert_array_equal(a.log_density,uninterrupted.posterior.log_density)
        final_gate=diagnose(uninterrupted,functional_draws=16,min_chains=4,
                            seed=seed_for(seed,"diagnostics"))
        assert final_gate==endpoint.diagnostics["gates"][-1]
