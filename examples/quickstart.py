"""Run from an installed source checkout: python examples/quickstart.py."""
from pathlib import Path
from bayes_stesn import (
    SpatioTemporalData, SplitPlan, ModelSpec, FeatureSpec, SpatialMeanSpec,
    PriorSpec, CovarianceSpec, SamplerConfig, prepare, fit, simulate_predictive,
    EvaluationTruth, score, diagnose,
)
from bayes_stesn.simulation import simulate_dataset

output=Path('results/quickstart');output.mkdir(parents=True,exist_ok=True)
simulation=simulate_dataset(800,grid_size=7,input_sizes=(25,),seed=42)
observed_sites=simulation.input_indices[25]
data=SpatioTemporalData(simulation.observation[:700,observed_sites],
                       simulation.coordinates[observed_sites],response_units='synthetic response')
spec=ModelSpec(
    features=FeatureSpec(modes=6,units=(16,16),squares=8),
    spatial_mean=SpatialMeanSpec(side=5), prior=PriorSpec.ridge(),
    covariance=CovarianceSpec(),
)
prepared=prepare(data,SplitPlan(conditioning=100,fitting_stop=700),spec)
fitted=fit(prepared,SamplerConfig(chains=4,warmup=800,draws=2000,seed=43),
           checkpoint_path=output/'checkpoint.npz')
prediction=simulate_predictive(fitted,horizon=40,target_coordinates=simulation.coordinates,
                                paths=128,seed=44)
truth=EvaluationTruth(simulation.observation[700:740],simulation.generating_mean[700:740],
                       simulation.latent[700:740])
scores=score(prediction,truth,target='observation',energy_pairs=None)
diagnostics=diagnose(fitted)
fitted.save(output/'model.npz');prediction.save(output/'forecast.npz')
print(f"Observation RMSE: {scores['rmse']:.4f}; CRPS: {scores['crps']:.4f}")
print(f"Scientific diagnostic gate passed: {diagnostics['passed']}")
print('A failed gate calls for investigation or longer sampling before interpreting performance.')

