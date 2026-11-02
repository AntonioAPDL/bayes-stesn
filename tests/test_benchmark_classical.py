"""Analytic references and information-contract checks for comparators."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from bayes_stesn.benchmark.contracts import Panel, CapabilityError
from bayes_stesn.benchmark.classical import (CoordinateBasis, fit_classical,
    load_classical, classical_from_state, matern_correlation)
from bayes_stesn.benchmark.gaussian_gp import temporal_system
from bayes_stesn.benchmark.linear_state import reduced_filter, rts_smooth


class TestSpatialBasisAndPointMethods(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(920)
        self.coordinates=self.rng.uniform(size=(10,2))
        t=np.arange(80)
        self.y=np.sin(2*np.pi*t[:,None]/12)+.1*self.rng.normal(size=(80,10))
        self.data=Panel(self.y,self.coordinates)

    def test_coordinate_basis_is_orthonormal_and_extends(self):
        basis=CoordinateBasis.fit(self.coordinates,rank=6,width=.3)
        phi=basis.evaluate(self.coordinates)
        np.testing.assert_allclose(phi.T @ phi/len(phi),np.eye(phi.shape[1]),atol=1e-11)
        self.assertEqual(basis.evaluate(np.array([[.4,.6]])).shape,(1,6))
        order=np.arange(10)[::-1]
        other=CoordinateBasis.fit(self.coordinates[order],rank=6,width=.3)
        # Eigenvector signs may differ; the function projection is invariant.
        np.testing.assert_allclose(other.evaluate(self.coordinates) @ other.evaluate(self.coordinates).T,
                                   phi @ phi.T,atol=1e-10)

    def test_persistence_and_period_wrap_without_future_truth(self):
        seasonal=fit_classical("seasonal_naive",self.data,{"period":12})
        prediction=seasonal.forecast(30)
        np.testing.assert_array_equal(prediction.mean,self.y[-12+np.arange(30)%12])
        self.assertIsNone(prediction.samples)
        with self.assertRaises(CapabilityError):
            prediction.interval()
        with self.assertRaises(CapabilityError):
            seasonal.forecast(2,target_coordinates=np.array([[2.,3.]]))
        persistence=fit_classical("persistence",self.data)
        np.testing.assert_array_equal(persistence.forecast(4).mean,np.tile(self.y[-1],(4,1)))

    def test_period_is_selected_using_training_and_can_fallback(self):
        clean=Panel(np.tile(np.sin(2*np.pi*np.arange(150)/12)[:,None],(1,10)),self.coordinates)
        model=fit_classical("seasonal_naive",clean)
        self.assertEqual(model.period,12)
        constant=fit_classical("seasonal_naive",Panel(np.ones((50,10)),self.coordinates))
        self.assertEqual(constant.period,1)

    def test_exact_and_full_landmark_kernel_ridge_agree(self):
        data=Panel(self.y[:35],self.coordinates)
        common={"rank":4,"lags":2,"ridge":.02,"bandwidth":3.,"nystrom_rank":33}
        exact=fit_classical("basis_krr",data,{**common,"exact":True},seed=2)
        approximate=fit_classical("basis_krr",data,{**common,"exact":False},seed=2)
        targets=np.array([[.2,.5],[.8,.5]])
        np.testing.assert_allclose(exact.forecast(6,targets).mean,approximate.forecast(6,targets).mean,atol=2e-8,rtol=2e-8)
        self.assertEqual(approximate.diagnostics["n_training_pairs"],33)
        self.assertEqual(approximate.diagnostics["library_ridge_equivalent"],33*.02)

    def test_history_must_extend_identical_observed_prefix(self):
        model=fit_classical("basis_krr",self.data.historical(60),{"rank":4,"nystrom_rank":24})
        original=model.forecast(4,history=self.data).mean
        np.testing.assert_array_equal(model.condition(self.data).forecast(4).mean,original)
        poison=self.y.copy();poison[3,1]+=1
        with self.assertRaises(ValueError):
            model.forecast(4,history=Panel(poison,self.coordinates))

    def test_point_spatial_extension_and_memory_gate(self):
        model=fit_classical("spatial_persistence",self.data)
        result=model.forecast(5,np.array([[.25,.75],[.75,.25]]))
        np.testing.assert_array_equal(result.mean,np.tile(result.mean[0],(5,1)))
        self.assertIsNone(result.samples)
        with self.assertRaises(MemoryError):
            fit_classical("basis_krr",self.data,{"exact":True,"max_exact_cases":10})


class TestGaussianProcessReference(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(812)
        self.coords=np.array([[0.,0.],[1.,0.],[0.,1.]])
        self.times=.7*np.arange(9)
        self.y=self.rng.normal(size=(9,3))
        self.data=Panel(self.y,self.coords,self.times)
        self.targets=np.array([[1.,0.],[.4,.6]])
        self.config={"fit_hyperparameters":False,"center":0.,"spatial_length":.45,
                     "temporal_length":2.3,"signal_sd":1.2,"noise_sd":.18}

    def dense_reference(self,nu,h=3):
        c=self.config;future=self.times[-1]+.7*np.arange(1,h+1)
        ks=matern_correlation(self.coords,self.coords,c["spatial_length"],1.5)
        kss=matern_correlation(self.targets,self.targets,c["spatial_length"],1.5)
        cross=matern_correlation(self.targets,self.coords,c["spatial_length"],1.5)
        kt=matern_correlation(self.times[:,None],self.times[:,None],c["temporal_length"],nu)
        ktt=matern_correlation(future[:,None],future[:,None],c["temporal_length"],nu)
        ktcross=matern_correlation(future[:,None],self.times[:,None],c["temporal_length"],nu)
        observed=c["signal_sd"]**2*np.kron(kt,ks)+c["noise_sd"]**2*np.eye(self.y.size)
        jointcross=c["signal_sd"]**2*np.kron(ktcross,cross)
        futurecov=c["signal_sd"]**2*np.kron(ktt,kss)+c["noise_sd"]**2*np.eye(h*len(self.targets))
        factor=cho_factor(observed,lower=True)
        mean=jointcross @ cho_solve(factor,self.y.ravel())
        cov=futurecov-jointcross @ cho_solve(factor,jointcross.T)
        ll=-.5*(self.y.size*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+self.y.ravel() @ cho_solve(factor,self.y.ravel()))
        return mean.reshape(h,-1),cov,ll

    def test_temporal_state_covariance_equals_matern(self):
        for nu in (.5,1.5):
            f,p,q=temporal_system(2.3,nu,.7)
            np.testing.assert_allclose(p,f @ p @ f.T+q,atol=1e-14)
            for lag in range(5):
                actual=(np.linalg.matrix_power(f,lag) @ p)[0,0]
                expected=matern_correlation(np.array([[0.]]),np.array([[lag*.7]]),2.3,nu)[0,0]
                self.assertAlmostEqual(actual,expected,places=12)

    def test_filter_and_joint_newsite_covariance_match_dense_gp(self):
        for nu in (.5,1.5):
            with self.subTest(nu=nu):
                model=fit_classical("separable_stgp",self.data,{**self.config,"temporal_nu":nu})
                expected_mean,expected_cov,expected_ll=self.dense_reference(nu)
                mean,cov=model.joint_moments(3,self.targets)
                np.testing.assert_allclose(mean,expected_mean,atol=2e-10)
                np.testing.assert_allclose(cov,expected_cov,atol=2e-10)
                self.assertAlmostEqual(model.diagnostics["log_likelihood"],expected_ll,places=9)
                np.testing.assert_allclose(model.forecast(3,self.targets,n_paths=0).mean,mean,atol=1e-12)

    def test_simulated_paths_retain_newsite_time_dependence(self):
        model=fit_classical("separable_stgp",self.data,{**self.config,"temporal_nu":1.5})
        mean,cov=model.joint_moments(3,self.targets)
        sample=model.forecast(3,self.targets,n_paths=6000,seed=88).samples.reshape(6000,-1)
        np.testing.assert_allclose(sample.mean(axis=0),mean.ravel(),atol=.035)
        np.testing.assert_allclose(np.cov(sample,rowvar=False),cov,atol=.045,rtol=.08)
        self.assertGreater(cov[1,3],.1)

    def test_path_and_horizon_prefix_and_duplicate_coordinates(self):
        model=fit_classical("separable_stgp",self.data,self.config)
        targets=np.vstack([self.targets,self.targets[1]])
        a=model.forecast(4,targets,n_paths=4,seed=81)
        b=model.forecast(6,targets,n_paths=7,seed=81)
        np.testing.assert_array_equal(a.samples,b.samples[:4,:4])
        np.testing.assert_array_equal(a.samples[:,:,1],a.samples[:,:,2])

    def test_frozen_parameter_conditioning_equals_full_refilter(self):
        more=Panel(np.vstack([self.y,self.rng.normal(size=(3,3))]),self.coords,.7*np.arange(12))
        first=fit_classical("separable_stgp",self.data,{**self.config,"temporal_nu":1.5})
        second=fit_classical("separable_stgp",more,{**self.config,"temporal_nu":1.5})
        a,ac=first.joint_moments(2,self.targets,history=more)
        b,bc=second.joint_moments(2,self.targets)
        np.testing.assert_allclose(a,b,atol=1e-11)
        np.testing.assert_allclose(ac,bc,atol=1e-11)


class TestLinearStateReference(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(102)
        self.n,self.r,self.t=5,2,7
        self.phi=np.linalg.qr(self.rng.normal(size=(self.n,self.r)))[0]*np.sqrt(self.n)
        self.a=np.array([[.85,.07],[-.08,.7]])
        self.q=np.array([.04,.025]);self.noise=.12
        self.y=self.rng.normal(size=(self.t,self.n))

    def test_reduced_filter_matches_full_observation_kalman(self):
        b=self.y @ self.phi/self.n
        m,p,ll,_=reduced_filter(b,self.a,self.q,self.noise,self.n)
        full_m=np.zeros(self.r);full_p=np.eye(self.r);full_ll=0.
        for t,y in enumerate(self.y):
            if t:
                full_m=self.a @ full_m;full_p=self.a @ full_p @ self.a.T+np.diag(self.q)
            s=self.phi @ full_p @ self.phi.T+self.noise*np.eye(self.n)
            factor=cho_factor(s,lower=True);innovation=y-self.phi @ full_m
            gain=cho_solve(factor,self.phi @ full_p).T
            full_m+=gain @ innovation;full_p-=gain @ self.phi @ full_p
            full_ll-=.5*(self.n*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()+innovation @ cho_solve(factor,innovation))
        correction=-.5*self.t*self.r*np.log(self.n)-.5*(self.t*(self.n-self.r)*np.log(2*np.pi*self.noise)+(np.sum(self.y**2)-self.n*np.sum(b**2))/self.noise)
        np.testing.assert_allclose(m,full_m,atol=1e-12)
        np.testing.assert_allclose(p,full_p,atol=1e-12)
        self.assertAlmostEqual(ll+correction,full_ll,places=10)

    def test_smoother_and_lag_covariances_match_dense_conditional(self):
        prior=np.zeros((self.t*self.r,self.t*self.r));prior[:self.r,:self.r]=np.eye(self.r)
        for t in range(1,self.t):
            last=slice((t-1)*self.r,t*self.r);now=slice(t*self.r,(t+1)*self.r)
            prior[now,now]=self.a @ prior[last,last] @ self.a.T+np.diag(self.q)
            for j in range(t):
                block=slice(j*self.r,(j+1)*self.r)
                prior[now,block]=self.a @ prior[last,block]
                prior[block,now]=prior[now,block].T
        loading=np.kron(np.eye(self.t),self.phi)
        cross=prior @ loading.T
        observed=loading @ cross+self.noise*np.eye(self.t*self.n)
        factor=cho_factor(observed,lower=True)
        mean=(cross @ cho_solve(factor,self.y.ravel())).reshape(self.t,self.r)
        cov=prior-cross @ cho_solve(factor,cross.T)
        _,_,_,history=reduced_filter(self.y @ self.phi/self.n,self.a,self.q,self.noise,self.n,store=True)
        ms,ps,lag=rts_smooth(history,self.a)
        np.testing.assert_allclose(ms,mean,atol=1e-11)
        for t in range(self.t):
            block=slice(t*self.r,(t+1)*self.r)
            np.testing.assert_allclose(ps[t],cov[block,block],atol=1e-11)
            if t:
                np.testing.assert_allclose(lag[t-1],cov[block,slice((t-1)*self.r,t*self.r)],atol=1e-11)

    def test_generalized_em_is_monotone_stable_and_forecasts_new_sites(self):
        coords=self.rng.uniform(size=(12,2));phi=CoordinateBasis.fit(coords,4,.3).evaluate(coords)
        states=np.empty((150,4));states[0]=self.rng.normal(size=4)
        for t in range(1,len(states)):
            states[t]=.8*states[t-1]+.18*self.rng.normal(size=4)
        data=Panel(states @ phi.T+.12*self.rng.normal(size=(150,12)),coords)
        model=fit_classical("basis_lgssm",data,{"rank":4,"basis_width":.3,"max_iter":100})
        self.assertTrue(model.diagnostics["converged"])
        self.assertTrue(model.diagnostics["qualified"])
        self.assertGreaterEqual(min(np.diff(model.diagnostics["log_likelihood"])),-1e-7)
        self.assertLessEqual(model.diagnostics["transition_spectral_norm"],.995+1e-10)
        targets=np.array([[.2,.8],[.8,.2],[.2,.8]])
        a=model.forecast(3,targets,n_paths=6000,seed=8)
        np.testing.assert_allclose(a.samples.mean(axis=0),a.mean,atol=.03)
        np.testing.assert_array_equal(a.samples[:,:,0],a.samples[:,:,2])
        b=model.forecast(5,targets,n_paths=6002,seed=8)
        np.testing.assert_array_equal(a.samples,b.samples[:6000,:3])


class TestPortableClassicalModels(unittest.TestCase):
    def test_each_classical_model_rejects_a_misspelled_option(self):
        data=Panel(np.arange(30,dtype=float).reshape(10,3),np.array([[0.,0.],[1.,0.],[0.,1.]]))
        misspellings={"persistence":"periodd", "seasonal_naive":"peroid",
                      "spatial_persistence":"spacial_length", "basis_krr":"nystrom_rnak",
                      "basis_lgssm":"max_itre", "separable_stgp":"tempral_nu"}
        for model_id,option in misspellings.items():
            with self.subTest(model_id=model_id):
                with self.assertRaisesRegex(ValueError,"Unknown .* configuration"):
                    fit_classical(model_id,data,{option:2})

    def test_integer_options_are_not_silently_truncated_or_boolean(self):
        data=Panel(np.arange(30,dtype=float).reshape(10,3),np.array([[0.,0.],[1.,0.],[0.,1.]]))
        invalid=[("seasonal_naive",{"period":True}),
                 ("seasonal_naive",{"min_period":0}),
                 ("basis_krr",{"lags":2.5}),
                 ("basis_krr",{"nystrom_rank":0}),
                 ("basis_lgssm",{"mstep_steps":0}),
                 ("basis_lgssm",{"rank":True}),
                 ("separable_stgp",{"max_iter":0}),
                 ("separable_stgp",{"max_forecast_bytes":-1})]
        for model_id,config in invalid:
            with self.subTest(model_id=model_id,config=config):
                with self.assertRaisesRegex(ValueError,"positive integer"):
                    fit_classical(model_id,data,config)

    def test_numeric_roundtrip_all_model_types(self):
        rng=np.random.default_rng(51);coords=rng.uniform(size=(8,2))
        data=Panel(rng.normal(size=(45,8)),coords)
        recipes={"persistence":{},"seasonal_naive":{"period":5},"spatial_persistence":{},
                 "basis_krr":{"rank":3,"nystrom_rank":16},
                 "basis_lgssm":{"rank":3,"max_iter":10},
                 "separable_stgp":{"fit_hyperparameters":False,"temporal_nu":1.5}}
        with tempfile.TemporaryDirectory() as tmp:
            for name,config in recipes.items():
                with self.subTest(name=name):
                    model=fit_classical(name,data,config,seed=8)
                    path=Path(tmp)/(name+".npz");model.save(path)
                    restored=load_classical(path)
                    a=model.forecast(4,n_paths=8,seed=12)
                    b=restored.forecast(4,n_paths=8,seed=12)
                    np.testing.assert_array_equal(a.mean,b.mean)
                    if a.samples is not None:
                        np.testing.assert_array_equal(a.samples,b.samples)
                    self.assertEqual(model.diagnostics,restored.diagnostics)
        state=fit_classical("persistence",data).to_state();state["version"]=99
        with self.assertRaises(ValueError):
            classical_from_state(state)


if __name__=="__main__":
    unittest.main()
