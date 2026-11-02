"""End-to-end coverage for registered extension routes and composed workflows.

The tiny chains here verify execution, identity and shape contracts. They are
deliberately not used as evidence that a scientific MCMC fit has converged.
"""
import dataclasses
import tempfile
import unittest
from pathlib import Path
import numpy as np
from bayes_stesn import *
from bayes_stesn.simulation import simulate_dataset
from bayes_stesn.workflow import fit_stacking_weights


class ExtensionWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.synthetic=simulate_dataset(70,grid_size=4,input_sizes=(9,),seed=98)
        cls.idx=cls.synthetic.input_indices[9]
        cls.data=SpatioTemporalData(cls.synthetic.observation[:45,cls.idx],cls.synthetic.coordinates[cls.idx])
        cls.base=ModelSpec(features=FeatureSpec(modes=2,units=(3,),input_scales=(.4,),squares=1),
                           spatial_mean=SpatialMeanSpec(side=2))
        cls.split=SplitPlan(12,45,validation_blocks=((45,50),),
                            calibration_blocks=((50,55),(55,60)),final_blocks=((60,65),))
        cls.config=SamplerConfig(chains=2,warmup=3,draws=4,seed=98)

    def check_route(self,spec,*,variance=False):
        F=np.sin(np.arange(45)*.1)[:,None,None]*(1+self.data.coordinates[None,:,0,None]) if variance else None
        prepared=prepare(self.data,self.split,spec,variance_covariates=F)
        fitted=fit(prepared,self.config)
        extra={}
        if variance:
            joint,_=joint_domain(self.data.coordinates,self.synthetic.coordinates)
            extra['future_variance_covariates']=np.sin(np.arange(45,48)*.1)[:,None,None]*(1+joint[None,:,0,None])
        sample=simulate_predictive(fitted,horizon=3,target_coordinates=self.synthetic.coordinates,paths=3,seed=11,**extra)
        self.assertEqual(sample.observation.shape,(3,3,16))
        self.assertTrue(np.isfinite(sample.observation).all())
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model.npz';fitted.save(path)
            restored=FittedModel.load(path)
            replay=simulate_predictive(restored,horizon=3,target_coordinates=self.synthetic.coordinates,paths=3,seed=11,**extra)
            np.testing.assert_array_equal(sample.observation,replay.observation)
        return fitted

    def test_all_prior_extensions_through_public_api(self):
        priors=[PriorSpec.grouped(),PriorSpec.group_rhs(),PriorSpec.column_rhs(),
                PriorSpec.entry_rhs(),PriorSpec.gigg(),PriorSpec.reference_r2(),
                PriorSpec.reference_r2(allocation='logistic_normal'),
                PriorSpec.correction(),PriorSpec.function_penalty()]
        for prior in priors:
            with self.subTest(prior=prior.kind,unit=prior.unit,allocation=prior.allocation):
                self.check_route(dataclasses.replace(self.base,prior=prior))

    def test_covariance_and_sparse_reference_extensions(self):
        covariance=[CovarianceSpec(anisotropy=True),CovarianceSpec(static_variance=True),
                    CovarianceSpec(infer_nu=True),CovarianceSpec(nonstationary=True,geometry_terms=1),
                    CovarianceSpec(range_prior='pc'),CovarianceSpec(range_prior='pc',scale_mode='separate'),
                    CovarianceSpec(kind='vecchia',vecchia_neighbors=4),
                    CovarianceSpec(kind='spde',mesh_resolution=4)]
        for cov in covariance:
            with self.subTest(cov=cov):
                self.check_route(dataclasses.replace(self.base,covariance=cov))

    def test_dynamic_robust_and_nonseparable_variance_extensions(self):
        for kind in ('full_residual_ar','latent_ar','field_t','local_t','heteroskedastic'):
            with self.subTest(kind=kind):
                self.check_route(dataclasses.replace(self.base,innovations=kind),variance=kind=='heteroskedastic')

    def test_augmented_checkpoint_extension_from_public_api(self):
        prepared=prepare(self.data,self.split,dataclasses.replace(self.base,innovations='latent_ar'))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'check.npz'
            first=dataclasses.replace(self.config,checkpoint_every=2)
            fit(prepared,first,checkpoint_path=path)
            longer=dataclasses.replace(first,draws=7)
            resumed=fit(prepared,longer,checkpoint_path=path,resume=True)
            full=fit(prepared,longer)
            np.testing.assert_array_equal(resumed.posterior.chains,full.posterior.chains)

    def test_ensemble_stacking_calibration_and_serialization(self):
        p=prepare(self.data,self.split,self.base)
        other=prepare(self.data,self.split,dataclasses.replace(self.base,
            features=dataclasses.replace(self.base.features,seed=732)))
        ensemble=fit_ensemble((p,other),self.config)
        predictions=[simulate_predictive(f,horizon=5,paths=12,seed=4) for f in ensemble.components]
        weighted=fit_stacking_weights(ensemble,predictions,self.synthetic.observation[45:50,self.idx])
        self.assertAlmostEqual(float(weighted.weights.sum()),1.)
        altered=dict(predictions[0].metadata);altered['parameter_posterior_identity']='different'
        wrong=dataclasses.replace(predictions[0],metadata=altered)
        with self.assertRaisesRegex(ValueError,'different component posterior'):
            fit_stacking_weights(ensemble,[wrong,predictions[1]],self.synthetic.observation[45:50,self.idx])
        mixed=simulate_predictive(weighted,horizon=5,paths=8,seed=7)
        self.assertEqual(len(mixed.metadata['mixture_component_id_per_path']),8)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'ensemble.npz';weighted.save(path)
            restored=FittedEnsemble.load(path)
            replay=simulate_predictive(restored,horizon=5,paths=8,seed=7)
            np.testing.assert_array_equal(mixed.observation,replay.observation)
        fitted=ensemble.components[0]
        calibration_predictions=[]
        for origin in (49,54):
            history=SpatioTemporalData(self.synthetic.observation[:origin+1,self.idx],self.data.coordinates)
            context=forecast_context(fitted,origin,history=history)
            calibration_predictions.append(simulate_predictive(fitted,context,horizon=5,paths=12,seed=origin))
        observed=np.stack([self.synthetic.observation[50:55,self.idx],self.synthetic.observation[55:60,self.idx]])
        calibration=fit_calibration(fitted,calibration_predictions,observed,spec=CalibrationSpec(target='simultaneous'))
        self.assertTrue(np.isinf(calibration.quantiles).all())
        with self.assertRaisesRegex(ValueError,'declared calibration'):
            fit_calibration(fitted,predictions[:1],observed[:1])
        with self.assertRaisesRegex(ValueError,'available at the forecast origin'):
            apply_calibration(calibration,calibration_predictions[0])
        history=SpatioTemporalData(self.synthetic.observation[:60,self.idx],self.data.coordinates)
        final=simulate_predictive(fitted,forecast_context(fitted,59,history=history),horizon=5,paths=12,seed=60)
        original=final.observation.copy()
        apply_calibration(calibration,final)
        np.testing.assert_array_equal(original,final.observation)

    def test_connected_bayesian_reservoir_index(self):
        p=prepare(self.data,self.split,self.base)
        q=prepare(self.data,self.split,dataclasses.replace(self.base,features=dataclasses.replace(self.base.features,seed=813)))
        ensemble=fit_reservoir_index((p,q),dataclasses.replace(self.config,warmup=8,draws=12))
        self.assertAlmostEqual(float(ensemble.weights.sum()),1.)
        sample=simulate_predictive(ensemble,horizon=3,paths=5,seed=1)
        self.assertEqual(sample.observation.shape,(5,3,9))
        self.assertEqual(ensemble.metadata['index_trace'].shape,(2,12))


if __name__=='__main__':
    unittest.main()
