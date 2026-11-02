"""Command-line entry points; configurations are plain, strictly parsed YAML."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
import yaml
from .model import ModelSpec, FittedModel, prepare, fit
from .data import SpatioTemporalData, SplitPlan
from .sampling import SamplerConfig
from .prediction import simulate_predictive
from .assessment import diagnose
from .scoring import score_predictions
from .simulation import simulate_dataset
from .io import atomic_npz, environment


def _json(value):
    if isinstance(value,np.ndarray):
        return _json(value.tolist())
    if isinstance(value,np.generic):
        return _json(value.item())
    if isinstance(value,dict):
        return {str(k):_json(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [_json(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):
        return None
    return value


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(_json(value),indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')


def load_config(path):
    config=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(config,dict):
        raise ValueError("Configuration must be a YAML mapping")
    unexpected=set(config)-{'schema_version','simulation','model','split','sampler','forecast'}
    if unexpected:
        raise ValueError(f"Unknown configuration sections: {sorted(unexpected)}")
    if config.get('schema_version',1)!=1:
        raise ValueError("Unsupported configuration schema")
    return config


def _run_demo(args):
    if args.config:
        config=load_config(args.config)
    else:
        config={'schema_version':1,'simulation':{'n_times':780,'grid_size':7,'input_sizes':[25],'seed':405},
            'model':{'features':{'modes':6,'units':[16,16],'input_scales':[.4,.8],'lags':[1,2], 'squares':8},
                     'spatial_mean':{'side':5,'width':.2}},
            'split':{'conditioning':100,'fitting_stop':700},
            'sampler':{'chains':4,'warmup':500,'draws':1000,'seed':406},
            'forecast':{'horizon':40,'paths':128,'seed':407}}
    if args.smoke:
        config={'schema_version':1,'simulation':{'n_times':150,'grid_size':5,'input_sizes':[12],'seed':405},
            'model':{'features':{'modes':3,'units':[6],'input_scales':[.4],'squares':3},
                     'spatial_mean':{'side':3,'width':.25}},
            'split':{'conditioning':30,'fitting_stop':130},
            'sampler':{'chains':2,'warmup':20,'draws':30,'seed':406},
            'forecast':{'horizon':10,'paths':16,'seed':407}}
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    (output/'resolved_configuration.yaml').write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
    synthetic=simulate_dataset(**config['simulation'])
    indices=synthetic.input_indices[max(synthetic.input_indices)]
    split=SplitPlan(**config['split'])
    data=SpatioTemporalData(synthetic.observation[:split.fitting_stop,indices],synthetic.coordinates[indices],
                            response_units='synthetic response')
    prepared=prepare(data,split,ModelSpec.from_dict(config.get('model',{})))
    fitted=fit(prepared,SamplerConfig(**config.get('sampler',{})),checkpoint_path=output/'checkpoint.npz')
    fitted.save(output/'model.npz')
    forecast=config.get('forecast',{'horizon':40,'paths':128,'seed':407})
    prediction=simulate_predictive(fitted,target_coordinates=synthetic.coordinates,**forecast)
    prediction.save(output/'prediction.npz')
    start=split.fitting_stop;stop=start+forecast['horizon']
    if stop>len(synthetic.observation):
        raise ValueError("Simulation needs to cover the entire forecast horizon")
    atomic_npz(output/'evaluation_truth.npz',observation=synthetic.observation[start:stop],
               generating_mean=synthetic.generating_mean[start:stop],latent=synthetic.latent[start:stop],
               coordinates=synthetic.coordinates,input_indices=indices)
    scores=score_predictions(prediction.observation,synthetic.observation[start:stop],coordinates=synthetic.coordinates,energy_pairs=None)
    gate=diagnose(fitted,functional_draws=min(max(8,fitted.posterior.n_draws),256))
    summary={'model':prepared.spec.to_dict(),'dimensions':{'sites':len(indices),'fitted_times':prepared.Y.shape[1],
               'spatial_basis':prepared.Phi.shape[1],'temporal_features':prepared.Z.shape[1]},
             'diagnostics':gate,'scores':scores,'metadata':fitted.metadata,
             'smoke_only':bool(args.smoke),'environment':environment(),
             'meaning':'Smoke demonstrations verify execution; consult the diagnostics before interpreting a fit.'}
    write_json(output/'summary.json',summary)
    print(json.dumps({'output':str(output.resolve()),'diagnostic_gate_passed':gate['passed'],
                      'rmse':scores['rmse'],'crps':scores['crps'],'smoke_only':bool(args.smoke)},indent=2))


def main(argv=None):
    parser=argparse.ArgumentParser(prog='bstesn',description='Bayesian spatiotemporal echo state networks')
    sub=parser.add_subparsers(dest='command',required=True)
    demo=sub.add_parser('demo',help='Run a fully reproducible synthetic-data fit and forecast')
    demo.add_argument('--config');demo.add_argument('--output',default='results/demo')
    demo.add_argument('--smoke',action='store_true',help='Small execution test, intentionally not a converged scientific fit')
    fit_command=sub.add_parser('fit',help='Fit observed y/coordinates/times arrays from a numeric NPZ file')
    fit_command.add_argument('data');fit_command.add_argument('--config',required=True)
    fit_command.add_argument('--output',required=True);fit_command.add_argument('--checkpoint')
    fit_command.add_argument('--resume',action='store_true')
    forecast=sub.add_parser('forecast',help='Forecast from a saved model at its fitting endpoint')
    forecast.add_argument('model');forecast.add_argument('--horizon',type=int,required=True)
    forecast.add_argument('--paths',type=int,default=256);forecast.add_argument('--seed',type=int,default=20261102)
    forecast.add_argument('--coordinates',help='Optional numeric .npy target coordinates')
    forecast.add_argument('--output',required=True)
    diagnostics=sub.add_parser('diagnose',help='Check scalar and readout functional chains')
    diagnostics.add_argument('model');diagnostics.add_argument('--functional-draws',type=int,default=256)
    diagnostics.add_argument('--output')
    validate=sub.add_parser('validate-config',help='Parse a configuration and validate all declared specs')
    validate.add_argument('config')
    args=parser.parse_args(argv)
    try:
        if args.command=='demo':
            _run_demo(args)
        elif args.command=='fit':
            config=load_config(args.config)
            with np.load(args.data,allow_pickle=False) as arrays:
                data=SpatioTemporalData(arrays['y'],arrays['coordinates'],arrays['times'] if 'times' in arrays else None,
                    tuple(arrays['site_ids'].tolist()) if 'site_ids' in arrays else None)
                F=arrays['variance_covariates'].copy() if 'variance_covariates' in arrays else None
            prepared=prepare(data,SplitPlan(**config['split']),ModelSpec.from_dict(config.get('model',{})),variance_covariates=F)
            fitted=fit(prepared,SamplerConfig(**config.get('sampler',{})),checkpoint_path=args.checkpoint,resume=args.resume)
            fitted.save(args.output)
            print(json.dumps({'model':str(Path(args.output).resolve()),'posterior_identity':fitted.posterior_identity}))
        elif args.command=='forecast':
            fitted=FittedModel.load(args.model)
            coordinates=None if args.coordinates is None else np.load(args.coordinates,allow_pickle=False)
            prediction=simulate_predictive(fitted,horizon=args.horizon,paths=args.paths,seed=args.seed,target_coordinates=coordinates)
            prediction.save(args.output)
            print(json.dumps({'prediction':str(Path(args.output).resolve()),'shape':list(prediction.observation.shape)}))
        elif args.command=='diagnose':
            result=diagnose(FittedModel.load(args.model),functional_draws=args.functional_draws)
            if args.output:
                write_json(args.output,result)
            print(json.dumps(_json(result),indent=2,allow_nan=False))
            return 0 if result['passed'] else 2
        else:
            config=load_config(args.config)
            ModelSpec.from_dict(config.get('model',{}));SplitPlan(**config['split']);SamplerConfig(**config.get('sampler',{}))
            print('Configuration valid; dimension-dependent compatibility is checked by prepare/fit.')
        return 0
    except (ValueError,TypeError,FileNotFoundError,KeyError) as error:
        parser.error(str(error))


if __name__=='__main__':
    sys.exit(main())

