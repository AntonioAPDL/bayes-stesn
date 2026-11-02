"""Optional neural control: joint spatial laws, recursion and portable state."""
import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np

from bayes_stesn.benchmark.contracts import Panel
from bayes_stesn.benchmark.classical import matern_correlation
from bayes_stesn.benchmark.basis_gru import (fit_basis_gru,basis_gru_from_state,
                                           load_basis_gru)


@unittest.skipUnless(importlib.util.find_spec("torch"),"optional PyTorch is unavailable")
class TestBasisGRU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(725)
        cls.coords=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.],[.4,.3],[.7,.8]])
        time=np.arange(100)
        signal=np.sin(.17*time[:,None]+cls.coords[:,0])+.3*np.cos(.09*time[:,None]-cls.coords[:,1])
        root=np.linalg.cholesky(.025*matern_correlation(cls.coords,cls.coords,.35)+.01*np.eye(6))
        cls.data=Panel(signal+rng.normal(size=signal.shape) @ root.T,cls.coords)
        cls.config={"rank":3,"lags":3,"hidden_size":8,"epochs":16,"patience":4,
                    "batch_size":32,"mean_paths":32,"prediction_batch_size":16,
                    "covariance_max_iter":80,"learning_rate":.01,"validation_horizon":12}
        cls.model=fit_basis_gru(cls.data,cls.config,seed=11)

    def test_finite_fit_and_honest_statistical_diagnostics(self):
        diagnostics=self.model.diagnostics
        self.assertTrue(diagnostics["qualified"])
        self.assertTrue(diagnostics["finite_fit"])
        self.assertEqual(diagnostics["neural_optimizer_convergence"],"not_claimed")
        self.assertFalse(diagnostics["parameter_uncertainty"])
        self.assertEqual(diagnostics["selection_training_rows"],80)
        self.assertEqual(diagnostics["covariance_fit_rows"],97)
        self.assertLessEqual(diagnostics["selected_epoch"],16)
        self.assertEqual(diagnostics["research_status"],"custom_matched_control_not_paper_reproduction")

    def test_first_step_matches_conditional_mean_and_dense_spatial_covariance(self):
        targets=np.array([[0.,0.],[.25,.65],[.9,.6]])
        forecast=self.model.forecast(1,targets,n_paths=7000,seed=81)
        phi=self.model.basis.evaluate(targets)
        history=((self.data.y[-3:]-self.model.center)/self.model.scale) @ self.model._phi/6
        expected=self.model.center+self.model.scale*(phi @ self.model._predict(history[None])[0])
        np.testing.assert_allclose(forecast.mean[0],expected,atol=1e-12)
        expected_covariance=self.model.innovation_covariance(targets)
        draws=forecast.samples[:,0]
        standard_errors=np.sqrt(np.diag(expected_covariance)/len(draws))
        self.assertTrue(np.all(np.abs(draws.mean(axis=0)-expected)<4.5*standard_errors))
        np.testing.assert_allclose(np.cov(draws,rowvar=False),expected_covariance,rtol=.07,atol=.015)

    def test_path_prefixes_duplicates_and_target_query_independence(self):
        targets=np.vstack([self.coords,np.array([[.25,.65],[.25,.65]])])
        small=self.model.forecast(3,targets,n_paths=4,seed=39)
        larger=self.model.forecast(5,targets,n_paths=43,seed=39)
        observed=self.model.forecast(3,n_paths=4,seed=39)
        point=self.model.forecast(3,targets,n_paths=0,seed=39)
        np.testing.assert_array_equal(small.samples,larger.samples[:4,:3])
        np.testing.assert_array_equal(small.samples[:,:,-2],small.samples[:,:,-1])
        np.testing.assert_array_equal(small.samples[:,:,:6],observed.samples)
        np.testing.assert_array_equal(small.mean,point.mean)
        np.testing.assert_array_equal(small.mean,larger.mean[:3])
        self.assertIsNone(point.samples)
        self.assertEqual(point.metadata["mean_paths"],32)
        self.assertEqual(point.metadata["newsite_history"],"never_used")

    def test_frozen_conditioning_and_memory_cap(self):
        rng=np.random.default_rng(33)
        extended=Panel(np.vstack([self.data.y,rng.normal(size=(3,6))]),self.coords)
        a=self.model.condition(extended).forecast(3,n_paths=4,seed=9)
        b=self.model.forecast(3,n_paths=4,seed=9,history=extended)
        np.testing.assert_array_equal(a.mean,b.mean)
        np.testing.assert_array_equal(a.samples,b.samples)
        corrupt=extended.y.copy();corrupt[2,0]+=1
        with self.assertRaises(ValueError):
            self.model.forecast(2,history=Panel(corrupt,self.coords))
        state=copy.deepcopy(self.model.to_state());state["config"]["max_domain_sites"]=6
        limited=basis_gru_from_state(state)
        with self.assertRaises(MemoryError):
            limited.forecast(2,np.array([[.25,.65]]))

    def test_portable_weights_and_rng_isolation(self):
        import torch
        rng_before=torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"gru.npz"
            self.model.save(path);restored=load_basis_gru(path)
            original=self.model.forecast(4,n_paths=5,seed=202)
            other=restored.forecast(4,n_paths=5,seed=202)
            np.testing.assert_array_equal(original.mean,other.mean)
            np.testing.assert_array_equal(original.samples,other.samples)
            self.assertEqual(restored.diagnostics,self.model.diagnostics)
        self.assertTrue(torch.equal(rng_before,torch.random.get_rng_state()))

    def test_selection_prefix_scales_and_deterministic_training(self):
        import torch
        config={**self.config,"epochs":2,"refit":False,"mean_paths":4}
        before=torch.random.get_rng_state().clone()
        first=fit_basis_gru(self.data,config,seed=40)
        second=fit_basis_gru(self.data,config,seed=40)
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        self.assertEqual(first.center,float(self.data.y[:80].mean()))
        self.assertEqual(first.scale,float(self.data.y[:80].std()))
        self.assertEqual(first.diagnostics["covariance_fit_rows"],77)
        for key,weights in first.to_state()["weights"].items():
            np.testing.assert_array_equal(weights,second.to_state()["weights"][key])
        np.testing.assert_array_equal(first.forecast(3,n_paths=4,seed=3).samples,
                                      second.forecast(3,n_paths=4,seed=3).samples)

    def test_unknown_configuration_and_short_history_fail(self):
        with self.assertRaises(ValueError):
            fit_basis_gru(self.data,{"epocs":3})
        with self.assertRaises(ValueError):
            fit_basis_gru(self.data.historical(10),{"lags":8})


if __name__=="__main__":
    unittest.main()
