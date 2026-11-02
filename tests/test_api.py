"""Information-policy, serialization and complete workflow regression tests."""
import dataclasses
import tempfile
import unittest
from pathlib import Path
import numpy as np

from bayes_stesn import (SpatioTemporalData, SplitPlan, FeatureSpec, SpatialMeanSpec,
    PriorSpec, CovarianceSpec, SamplerConfig, ModelSpec, FittedModel, prepare, fit,
    forecast_context, advance_state, simulate_predictive, PredictiveSample, joint_domain)
from bayes_stesn.features import FrozenFeatures
from bayes_stesn.basis import SpatialMean
from bayes_stesn.io import save_state, load_state, content_hash, seed_for


class TestDataFeatures(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(21)
        self.y = rng.normal(size=(100, 12))
        self.coordinates = rng.uniform(size=(12, 2))
        self.features = FeatureSpec(modes=3, units=(8, 6), squares=4)
        self.spec = ModelSpec(features=self.features, spatial_mean=SpatialMeanSpec(side=3))

    def test_invalid_data_fail_before_fit(self):
        y = self.y.copy(); y[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            SpatioTemporalData(y, self.coordinates)
        with self.assertRaisesRegex(ValueError, "equally spaced"):
            SpatioTemporalData(self.y, self.coordinates, times=np.arange(100)**2)
        with self.assertRaisesRegex(ValueError, "Project"):
            SpatioTemporalData(self.y, self.coordinates, coordinate_units="degrees")
        with self.assertRaisesRegex(ValueError, "unique"):
            SpatioTemporalData(self.y, np.zeros_like(self.coordinates))

    def test_arrays_are_owned_and_readonly(self):
        data = SpatioTemporalData(self.y, self.coordinates)
        self.y[0, 0] = 999
        self.assertNotEqual(data.y[0, 0], 999)
        with self.assertRaises(ValueError):
            data.y[0, 0] = 3

    def test_disjoint_split_required(self):
        with self.assertRaises(ValueError):
            SplitPlan(20, 60, validation_blocks=((55, 70),))
        with self.assertRaises(ValueError):
            SplitPlan(20, 60, validation_blocks=((60, 80),), final_blocks=((70, 90),))

    def test_sealed_future_does_not_change_prepared_identity(self):
        a = prepare(SpatioTemporalData(self.y, self.coordinates), SplitPlan(30, 80), self.spec)
        poisoned = self.y.copy(); poisoned[80:] = 1e5
        b = prepare(SpatioTemporalData(poisoned, self.coordinates), SplitPlan(30, 80), self.spec)
        self.assertEqual(a.identity, b.identity)
        np.testing.assert_array_equal(a.Z, b.Z)
        self.assertEqual(len(a.data.y), 80)

    def test_conditioning_only_and_one_step_causality(self):
        a = FrozenFeatures.fit(self.y, 30, self.coordinates, self.features)
        poison = self.y.copy(); poison[60:] += 1000
        b = FrozenFeatures.fit(poison, 30, self.coordinates, self.features)
        self.assertEqual(a.fingerprint, b.fingerprint)
        np.testing.assert_array_equal(a.Z[:61], b.Z[:61])
        self.assertGreater(np.max(np.abs(a.Z[61:]-b.Z[61:])), 0.01)

    def test_source_faithful_v3_fixture(self):
        path = Path(__file__).parent/'fixtures'/'v3_reference.npz'
        with np.load(path, allow_pickle=False) as f:
            spec = FeatureSpec(modes=3, units=(8,6), lags=(1,3), squares=4, products=3,
                               input_scales=(.4,.8), score_bound=3, seed=731)
            features = FrozenFeatures.fit(f['y'],30,f['coordinates'],spec)
            np.testing.assert_allclose(features.Z,f['Z'],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(features.states,f['states'],rtol=1e-12,atol=1e-12)
            basis = SpatialMean.fit(f['coordinates'], SpatialMeanSpec(side=3,width=.18,bounds=((0,1),(0,1))))
            np.testing.assert_allclose(basis.transform(f['coordinates']),f['phi'],rtol=1e-12,atol=1e-12)

    def test_spatial_and_learned_encoders_are_causal(self):
        for kind in ('local_features_esn','graph_features_esn','frozen_encoder_esn'):
            with self.subTest(kind=kind):
                spec = dataclasses.replace(self.features,kind=kind,encoder_width=6,encoder_epochs=40)
                first = FrozenFeatures.fit(self.y,30,self.coordinates,spec)
                changed = self.y.copy(); changed[40:] *= 100
                second = FrozenFeatures.fit(changed,30,self.coordinates,spec)
                self.assertEqual(first.fingerprint,second.fingerprint)
                np.testing.assert_array_equal(first.Z[:41],second.Z[:41])
                if kind=='frozen_encoder_esn':
                    loss=first.encoder.parameters['training_loss']
                    self.assertLess(loss[-1],loss[0])

    def test_correction_is_orthogonal_on_reference(self):
        features = FrozenFeatures.fit(self.y,30,self.coordinates,self.features,correction_base=4)
        z=features.Z[10:30]
        np.testing.assert_allclose(z[:,:4].T@z[:,4:],0,atol=1e-11)

    def test_feature_save_roundtrip_and_batched_replay(self):
        f=FrozenFeatures.fit(self.y,30,self.coordinates,self.features)
        loaded=FrozenFeatures.from_state(f.to_state())
        np.testing.assert_array_equal(f.Z,loaded.Z)
        states,z=f.update_batch(self.y[40:44],f.states[40:44])
        for j in range(4):
            state,one=f.update(self.y[40+j],f.states[40+j])
            np.testing.assert_allclose(states[j],state,atol=1e-14)
            np.testing.assert_allclose(z[j],one,atol=1e-14)

    def test_portable_numeric_state_and_named_streams(self):
        state={'array':self.y,'tuple':(1,'x'),'mapping':{'a':np.float64(2)}}
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'state.npz'
            save_state(path,state)
            restored=load_state(path)
            self.assertEqual(content_hash(state),content_hash(restored))
            with self.assertRaises(TypeError):
                save_state(path,{'bad':np.array([{}],dtype=object)})
        self.assertEqual(seed_for(3,'path',8),seed_for(3,'path',8))
        self.assertNotEqual(seed_for(3,'path',8),seed_for(3,'path',9))


class TestCompleteWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from bayes_stesn.simulation import simulate_dataset
        cls.synthetic=simulate_dataset(160,grid_size=4,input_sizes=(9,),seed=88)
        cls.idx=cls.synthetic.input_indices[9]
        cls.data=SpatioTemporalData(cls.synthetic.observation[:130,cls.idx],cls.synthetic.coordinates[cls.idx])
        spec=ModelSpec(features=FeatureSpec(modes=3,units=(6,),input_scales=(.4,),squares=3),
                       spatial_mean=SpatialMeanSpec(side=3))
        cls.prepared=prepare(cls.data,SplitPlan(30,130),spec)
        cls.config=SamplerConfig(chains=2,warmup=30,draws=35,seed=98)
        cls.fitted=fit(cls.prepared,cls.config)

    def test_joint_duplicate_site_identity(self):
        coordinates=np.vstack([self.data.coordinates[2],self.synthetic.coordinates[0],self.data.coordinates[2]])
        sample=simulate_predictive(self.fitted,horizon=5,target_coordinates=coordinates,paths=12,seed=18)
        np.testing.assert_array_equal(sample.observation[:,:,0],sample.observation[:,:,2])
        with self.assertRaisesRegex(ValueError,'different input coordinate'):
            joint_domain(self.data.coordinates,1+self.data.coordinates[:1],input_ids=self.data.site_ids,
                         target_ids=(self.data.site_ids[0],))

    def test_same_seed_and_path_prefix(self):
        a=simulate_predictive(self.fitted,horizon=7,paths=8,seed=12)
        b=simulate_predictive(self.fitted,horizon=7,paths=10,seed=12)
        np.testing.assert_array_equal(a.observation,b.observation[:8])
        self.assertFalse(np.array_equal(a.observation,a.latent))
        self.assertEqual(a.metadata['forecast_feedback'],'simulated observations at the input sites')

    def test_save_load_reproduces_forecast(self):
        a=simulate_predictive(self.fitted,horizon=5,paths=6,seed=6)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.npz'; self.fitted.save(path)
            loaded=FittedModel.load(path)
            b=simulate_predictive(loaded,horizon=5,paths=6,seed=6)
            np.testing.assert_array_equal(a.observation,b.observation)
            path=Path(temp)/'prediction.npz'; a.save(path)
            np.testing.assert_array_equal(a.observation,PredictiveSample.load(path).observation)

    def test_advance_state_preserves_parameter_posterior(self):
        context=forecast_context(self.fitted)
        new=SpatioTemporalData(self.synthetic.observation[130:140,self.idx],self.data.coordinates,
                               times=np.arange(130,140),site_ids=self.data.site_ids)
        advanced=advance_state(self.fitted,context,new)
        self.assertEqual(advanced.parameter_posterior_identity,self.fitted.posterior_identity)
        self.assertEqual(advanced.origin,139)
        full=SpatioTemporalData(self.synthetic.observation[:140,self.idx],self.data.coordinates)
        replay=forecast_context(self.fitted,139,history=full)
        np.testing.assert_array_equal(advanced.Z,replay.Z)
        with self.assertRaises(ValueError):
            forecast_context(self.fitted,129,parameter_policy='refit')

    def test_wrong_posterior_and_insufficient_history_rejected(self):
        context=dataclasses.replace(forecast_context(self.fitted),parameter_posterior_identity='wrong')
        with self.assertRaises(ValueError):
            simulate_predictive(self.fitted,context,horizon=1,paths=1)
        with self.assertRaises(ValueError):
            forecast_context(self.fitted,140)

    def test_completed_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'checkpoint.npz'
            config=dataclasses.replace(self.config,warmup=5,draws=6,checkpoint_every=3)
            a=fit(self.prepared,config,checkpoint_path=path)
            b=fit(self.prepared,config,checkpoint_path=path,resume=True)
            np.testing.assert_array_equal(a.posterior.chains,b.posterior.chains)
            with self.assertRaisesRegex(ValueError,'identity mismatch'):
                fit(self.prepared,dataclasses.replace(config,seed=91),checkpoint_path=path,resume=True)

    def test_short_chains_do_not_claim_diagnostic_success(self):
        diagnostics=self.fitted.diagnose()
        self.assertFalse(diagnostics.passed)


if __name__=='__main__':
    unittest.main()
