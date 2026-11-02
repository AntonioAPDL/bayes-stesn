"""Analytic, information-boundary and numerical tests for shared field DGPs."""
import tempfile
from pathlib import Path
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import block_diag

from bayes_stesn.benchmark.generators import (
    Simulation, generate, make_design, grid_coordinates, matern32,
    spectral_spec, gaussian_ar_oracle_moments, fhn_current, fhn_equilibrium,
    volatility_reference,
)
from bayes_stesn.simulation import simulate_dataset


class GeometryTests(unittest.TestCase):
    def test_main_roles_and_fixed_density_targets(self):
        c=grid_coordinates(25)
        d=make_design(c,100,20)
        self.assertEqual(tuple(map(len,[d.observed_indices,d.calibration_indices,d.test_indices])),(100,20,505))
        self.assertEqual(len(set(np.r_[d.observed_indices,d.calibration_indices,d.test_indices])),625)
        self.assertGreater(len(d.hole_indices),0)
        curve=make_design(c,25,20,master_observed=200)
        self.assertEqual(len(curve.test_indices),405)
        for n in (25,50,100,200):
            same=curve.at_count(n)
            assert_array_equal(same.test_indices,curve.test_indices)
            assert_array_equal(same.observed_indices[:25],curve.observed_indices)
            self.assertFalse(set(same.observed_indices)&set(curve.test_indices))

    def test_full_grid_and_periodic_geometry(self):
        c=grid_coordinates(7,periodic=True)
        self.assertLess(c.max(),1)
        d=make_design(c,49,0)
        self.assertEqual(len(d.test_indices),0)
        self.assertEqual(len(d.observed_indices),49)
        with self.assertRaises(ValueError):
            generate("g2",5,coordinates=grid_coordinates(3),n_observed=3,n_calibration=0)


class GaussianTests(unittest.TestCase):
    def test_stationarity_and_lag_covariance(self):
        sim=generate("g1",12000,n_side=3,n_observed=3,n_calibration=1,seed=421,rho=.3)
        covariance=matern32(sim.coordinates)
        assert_allclose(np.cov(sim.latent.T,bias=True),covariance,atol=.065,rtol=0)
        lag=sim.latent[1:].T@sim.latent[:-1]/(len(sim.latent)-1)
        assert_allclose(lag,.3*covariance,atol=.055,rtol=0)
        noise=sim.observation-sim.latent
        assert_allclose(np.var(noise),.15**2,atol=.0004)

    def test_known_parameter_oracle_matches_joint_dense_conditioning(self):
        coords=np.array([[.1,.2],[.8,.7],[.5,.1]])
        obs=coords[:2];target=coords[1:];rho=.77;ell=.3;sd=.2
        history=np.array([[.2,-.8],[.3,-.4],[.1,-.2]])
        oracle=gaussian_ar_oracle_moments(history,obs,target,3,rho=rho,ell=ell,sigma_e=sd)
        times=np.arange(3);future=np.arange(3,6)
        koo=matern32(obs,ell=ell);kto=matern32(target,obs,ell=ell);ktt=matern32(target,ell=ell)
        cov_y=np.kron(rho**np.abs(times[:,None]-times),koo)+sd**2*np.eye(6)
        cross=np.kron(rho**np.abs(future[:,None]-times),kto)
        cov_future=np.kron(rho**np.abs(future[:,None]-future),ktt)+sd**2*np.eye(6)
        mean=cross@np.linalg.solve(cov_y,history.ravel())
        variance=np.diag(cov_future-cross@np.linalg.solve(cov_y,cross.T))
        assert_allclose(oracle["mean"],mean.reshape(3,2),atol=2e-13)
        assert_allclose(oracle["variance"],variance.reshape(3,2),atol=2e-13)


class SpectralTests(unittest.TestCase):
    def test_stationary_spectrum_and_transport_phase(self):
        spec=spectral_spec(5)
        self.assertAlmostEqual(spec["zero_power"]+2*sum(spec["power"]),1)
        self.assertTrue(np.all(np.abs(spec["multiplier"])<1))
        other=spectral_spec(5,velocity=(-.1,.3))
        assert_allclose(np.abs(other["multiplier"]),np.abs(spec["multiplier"]),atol=1e-15)
        h=spectral_spec(5,delta=.5)
        assert_allclose(h["multiplier"]**2,spec["multiplier"],atol=5e-16)

    def test_stationary_point_variance(self):
        sim=generate("g2",24000,n_side=3,n_observed=3,n_calibration=1,seed=75,
                     gamma=.4,kappa=.02,max_frequency=3,sigma_e=0)
        assert_allclose(np.var(sim.latent,axis=0),1,atol=.065,rtol=0)
        self.assertTrue(np.isrealobj(sim.latent))

    def test_shift_first_affected_row_and_equal_prior_prefix(self):
        opts=dict(n_side=5,n_observed=10,n_calibration=2,seed=117,max_frequency=3)
        base=generate("g2",70,**opts)
        changed=generate("g6",70,origin_index=40,break_lead=5,**opts)
        assert_array_equal(base.observation[:44],changed.observation[:44])
        self.assertGreater(np.max(np.abs(base.latent[44]-changed.latent[44])),1e-5)
        self.assertEqual(changed.metadata["parameters"]["break_index"],44)
        with self.assertRaises(ValueError):
            generate("g6",70,**opts)


class FrontAndStressTests(unittest.TestCase):
    def test_exact_existing_front_generator(self):
        sim=generate("g4",35,n_side=5,n_observed=10,n_calibration=2,seed=81)
        reference=simulate_dataset(35,coordinates=sim.coordinates,scenario="matched_gaussian",seed=81,
                                   stage="benchmark-fronts-v2",replication=0,input_sizes=(1,),burnin=500)
        assert_allclose(sim.observation,reference.observation,rtol=0,atol=2e-16)
        assert_array_equal(sim.generating_mean,reference.generating_mean)

    def test_stress_arms_change_only_declared_component(self):
        opts=dict(n_side=5,n_observed=10,n_calibration=2,seed=81)
        base=generate("g4",60,**opts)
        local=generate("g5",60,arm="local_t",**opts)
        field=generate("g5",60,arm="whole_field_t",**opts)
        hetero=generate("g5",60,arm="latent_state_heteroskedastic",**opts)
        assert_array_equal(local.latent,base.latent)
        assert_array_equal(hetero.latent,base.latent)
        assert_array_equal(field.generating_mean,base.generating_mean)
        assert_allclose(field.observation-field.latent,base.observation-base.latent,atol=3e-16,rtol=0)
        self.assertGreater(np.max(np.abs(field.latent-base.latent)),1e-4)
        self.assertNotEqual(float(local.observation.sum()),float(base.observation.sum()))
        self.assertGreater(volatility_reference(),0)


class FHNTests(unittest.TestCase):
    def test_equilibria_solve_kinetic_equations(self):
        current=np.array([.2,.48,.52])
        u,v=fhn_equilibrium(current)
        assert_allclose(u-u**3/3-v+current,0,atol=2e-15)
        trace=1-u*u-.08*.8
        self.assertLess(trace[0],0)
        self.assertTrue(np.all(trace[1:]>0))

    def test_step_refinement_and_hidden_state(self):
        opts=dict(n_side=5,n_observed=10,n_calibration=2,seed=42,grid_size=24,
                  burnin=2.,delta=.2,sigma_e=0)
        coarse=generate("g3",12,dt=.05,**opts)
        fine=generate("g3",12,dt=.025,**opts)
        self.assertIsNotNone(coarse.hidden_state)
        self.assertLess(float(np.sqrt(np.mean((coarse.latent-fine.latent)**2))),2e-4)
        self.assertLess(coarse.metadata["parameters"]["max_abs_state_at_observation_times"],10)
        self.assertFalse(hasattr(coarse.observed_panel(6),"hidden_state"))
        self.assertEqual(coarse.observed_panel(6).y.shape,(6,10))


class ContractTests(unittest.TestCase):
    def test_horizon_prefix_and_sensor_selection_do_not_change_data(self):
        for scenario in ("g1","g2","g4","g5"):
            with self.subTest(scenario=scenario):
                args=dict(n_side=5,n_calibration=2,seed=14)
                first=generate(scenario,20,n_observed=7,**args)
                extended=generate(scenario,35,n_observed=12,**args)
                assert_allclose(first.observation,extended.observation[:20],atol=5e-15,rtol=0)
                assert_array_equal(first.observed_indices,extended.observed_indices[:7])

    def test_roundtrip_and_isolated_visible_panel(self):
        sim=generate("g1",10,n_side=5,n_observed=7,n_calibration=2,seed=1)
        panel=sim.observed_panel(6)
        self.assertFalse(np.shares_memory(panel.y,sim.observation))
        self.assertFalse(hasattr(panel,"latent"))
        self.assertFalse(panel.y.flags.writeable)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"simulation.npz";sim.save(path)
            loaded=Simulation.load(path)
            assert_array_equal(loaded.observation,sim.observation)
            self.assertEqual(loaded.metadata,sim.metadata)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):generate("unknown",5)
        with self.assertRaises(ValueError):generate("g1",1)
        with self.assertRaises(ValueError):make_design(grid_coordinates(3),8,2)
        with self.assertRaises(ValueError):generate("g5",10,n_side=5,n_observed=7,n_calibration=2,nu=2)
        with self.assertRaises(ValueError):generate("g3",10,n_side=5,n_observed=7,n_calibration=2,dt=.07,delta=.2)


if __name__ == "__main__":
    unittest.main()
