"""Shared spatial-field DGPs with explicit hidden truth and information roles.

The simulators never depend on fitted-model features. Sensor selection is a
geometry-only operation on an already specified observation domain. Arrays
use [time, site]; coordinates exposed to models are normalized to [0, 1]^2.
FHN physical coordinates and integration units are recorded separately.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math

import numpy as np
from scipy.fft import dctn, idctn
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.ndimage import map_coordinates
from scipy.spatial.distance import cdist

from ..data import readonly
from ..io import seed_for, save_state, load_state
from ..simulation import simulate_dataset, phase_path, structured_mean, nested_input_design
from .contracts import Panel


SCENARIOS = ("g1_gaussian_ar", "g2_spectral_transport", "g3_excitable_field",
             "g4_existing_fronts", "g5_noise_stress", "g6_velocity_shift")
ALIASES = dict(zip(("g1", "g2", "g3", "g4", "g5", "g6"), SCENARIOS))
HOLE = (0.60, 0.84, 0.12, 0.36)


def _rng(seed, component):
    return np.random.default_rng(seed_for(seed, "benchmark-dgp-v2", component))


def _coordinates(value):
    value = np.asarray(value, float)
    if (value.ndim != 2 or value.shape[1] != 2 or len(value) < 1 or
            not np.isfinite(value).all() or np.any(value < 0) or np.any(value > 1) or
            len(np.unique(value, axis=0)) != len(value)):
        raise ValueError("coordinates must be distinct finite points in [0,1]^2")
    return value


def grid_coordinates(n_side=25, *, periodic=False):
    if not isinstance(n_side, (int, np.integer)) or n_side < 2:
        raise ValueError("n_side must be an integer >=2")
    axis = np.linspace(0, 1, int(n_side), endpoint=not periodic)
    return np.column_stack([np.tile(axis, n_side), np.repeat(axis, n_side)])


@dataclass(frozen=True)
class SpatialDesign:
    observed_indices: np.ndarray
    calibration_indices: np.ndarray
    test_indices: np.ndarray
    hole_indices: np.ndarray
    master_observed_indices: np.ndarray

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, readonly(getattr(self, name), int))

    @property
    def inactive_sensor_indices(self):
        return self.master_observed_indices[len(self.observed_indices):]

    def at_count(self, count):
        """Nested sensors, unchanged C/U targets and unchanged master domain."""
        if not 1 <= count <= len(self.master_observed_indices):
            raise ValueError("count must lie inside the frozen sensor master set")
        return SpatialDesign(self.master_observed_indices[:count], self.calibration_indices,
                             self.test_indices, self.hole_indices, self.master_observed_indices)


def make_design(coordinates, n_observed=100, n_calibration=20, *, seed=20261102,
                master_observed=None, hole=HOLE):
    """Freeze disjoint sensor, label-only calibration and final target roles.

    Setting master_observed>n_observed reserves the larger sensor population
    for a density curve; those inactive sensor sites never join scored U.
    Full-grid studies set n_observed=n_sites and n_calibration=0 explicitly.
    """
    coords = _coordinates(coordinates)
    master = n_observed if master_observed is None else master_observed
    if any(not isinstance(x, (int, np.integer)) for x in (n_observed, n_calibration, master)):
        raise ValueError("spatial role counts must be integers")
    if not 1 <= n_observed <= master <= len(coords) or n_calibration < 0 or master+n_calibration > len(coords):
        raise ValueError("invalid spatial role counts")
    x0, x1, y0, y1 = hole
    inside = ((coords[:, 0] >= x0) & (coords[:, 0] <= x1) &
              (coords[:, 1] >= y0) & (coords[:, 1] <= y1))
    if master == len(coords):
        order = np.arange(len(coords))
    else:
        order = nested_input_design(coords, [int(master)], seed=seed,
                                    excluded_rectangle=tuple(hole))[int(master)]
    remaining = np.setdiff1d(np.arange(len(coords)), order)
    rng = _rng(seed, "calibration-geometry")
    # Represent the hole while keeping at least one hole site sealed when possible.
    hole_remaining = remaining[inside[remaining]]
    outside_remaining = remaining[~inside[remaining]]
    desired = int(round(n_calibration*len(hole_remaining)/max(1, len(remaining))))
    if n_calibration and len(hole_remaining) >= 2:
        desired = max(1, desired)
    n_hole = min(desired, max(0, len(hole_remaining)-1), n_calibration)
    n_hole = max(n_hole, n_calibration-len(outside_remaining))
    cal = np.r_[rng.permutation(hole_remaining)[:n_hole],
                rng.permutation(outside_remaining)[:n_calibration-n_hole]].astype(int)
    test = np.setdiff1d(remaining, cal)
    return SpatialDesign(order[:n_observed], cal, test, test[inside[test]], order)


@dataclass(frozen=True)
class Simulation:
    """Evaluator-owned truth. Models receive only ``observed_panel`` copies."""
    observation: np.ndarray
    latent: np.ndarray
    coordinates: np.ndarray
    times: np.ndarray
    observed_indices: np.ndarray
    calibration_indices: np.ndarray
    test_indices: np.ndarray
    hole_indices: np.ndarray
    metadata: dict
    generating_mean: np.ndarray | None = None
    hidden_state: np.ndarray | None = None

    def __post_init__(self):
        y = np.asarray(self.observation, float)
        if y.ndim != 2 or min(y.shape) < 1 or not np.isfinite(y).all():
            raise ValueError("simulation observations must be finite [time,site]")
        coords = _coordinates(self.coordinates)
        if y.shape[1] != len(coords) or np.asarray(self.times).shape != (len(y),):
            raise ValueError("simulation axes do not align")
        for name in ("observation", "latent", "generating_mean", "hidden_state"):
            value = getattr(self, name)
            if value is not None:
                value = np.asarray(value)
                if value.shape != y.shape or not np.isfinite(value).all():
                    raise ValueError(f"{name} must be finite and match observations")
                object.__setattr__(self, name, readonly(value))
        for name in ("coordinates", "times"):
            object.__setattr__(self, name, readonly(getattr(self, name)))
        roles = []
        for name in ("observed_indices", "calibration_indices", "test_indices"):
            indices = np.asarray(getattr(self, name), int)
            if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= len(coords)):
                raise ValueError("invalid role indices")
            roles.extend(indices.tolist())
            object.__setattr__(self, name, readonly(indices, int))
        if len(set(roles)) != len(roles):
            raise ValueError("observed, calibration and final target roles must be disjoint")
        if not set(self.hole_indices).issubset(set(self.test_indices)):
            raise ValueError("hole targets must be a subset of final unseen sites")
        object.__setattr__(self, "hole_indices", readonly(self.hole_indices, int))
        object.__setattr__(self, "metadata", json.loads(json.dumps(self.metadata, allow_nan=False)))

    @property
    def site_ids(self):
        return tuple(f"site_{j:05d}" for j in range(len(self.coordinates)))

    @property
    def panel_all(self):
        """Evaluator-only full observations; never pass this directly to a fit."""
        return self.panel(np.arange(len(self.coordinates)))

    def panel(self, sites, stop=None):
        stop = len(self.observation) if stop is None else int(stop)
        if not 1 <= stop <= len(self.observation):
            raise ValueError("stop must identify a nonempty historical prefix")
        sites = np.asarray(sites, int)
        return Panel(self.observation[:stop, sites], self.coordinates[sites], self.times[:stop],
                     tuple(self.site_ids[j] for j in sites), "DGP response", "normalized domain",
                     {"information_policy": "explicit_visible_observations_only"})

    def observed_panel(self, stop=None):
        return self.panel(self.observed_indices, stop)

    def save(self, path):
        return save_state(path, self.__dict__, kind="benchmark_simulation")

    @classmethod
    def load(cls, path):
        return cls(**load_state(path, kind="benchmark_simulation"))


def matern32(coordinates, other=None, *, ell=.2):
    if not np.isfinite(ell) or ell <= 0:
        raise ValueError("ell must be positive")
    coords = _coordinates(coordinates)
    other = coords if other is None else _coordinates(other)
    radius = np.sqrt(3)*cdist(coords, other)/ell
    return (1+radius)*np.exp(-radius)


def _gaussian_ar(coords, n_times, seed, *, rho=.97, ell=.2, sigma_e=.15):
    if not abs(rho) < 1 or sigma_e < 0:
        raise ValueError("Gaussian AR requires |rho|<1 and nonnegative measurement sd")
    kernel = matern32(coords, ell=ell)
    root = np.linalg.cholesky(kernel)
    process = _rng(seed, "gaussian-ar-process").normal(size=(n_times, len(coords)))@root.T
    latent = process.copy()
    innovation_scale = np.sqrt(1-rho*rho)
    for t in range(1, n_times):
        latent[t] = rho*latent[t-1]+innovation_scale*process[t]
    observed = latent+sigma_e*_rng(seed, "measurement").normal(size=latent.shape)
    return observed, latent, None, None, {"rho":rho, "ell":ell, "sigma_e":sigma_e,
        "stationary_initialization":True, "matern_smoothness":1.5,
        "matern_convention":"(1+sqrt(3)*d/ell)*exp(-sqrt(3)*d/ell)", "covariance_jitter":0.0}


def spectral_spec(max_frequency=8, *, gamma=-math.log(.995), kappa=2e-4,
                  velocity=(.006,.003), delta=1.):
    """Independent half-plane modes and stationary powers for a real field."""
    if not isinstance(max_frequency, (int, np.integer)) or max_frequency < 0:
        raise ValueError("max_frequency must be a nonnegative integer")
    velocity = np.asarray(velocity, float)
    if gamma <= 0 or kappa < 0 or delta <= 0 or velocity.shape != (2,) or not np.isfinite(velocity).all():
        raise ValueError("invalid spectral parameters")
    k = np.array([(x,y) for x in range(-max_frequency,max_frequency+1)
                  for y in range(-max_frequency,max_frequency+1)
                  if x > 0 or (x == 0 and y > 0)], float).reshape(-1,2)
    power = (1+np.sum(k*k,axis=1))**-2.5
    normalization = 1+2*np.sum(power)
    power /= normalization
    multiplier = np.exp(-(gamma+4*np.pi**2*kappa*np.sum(k*k,axis=1))*delta)*np.exp(-2j*np.pi*(k@velocity)*delta)
    return {"wavevectors":k,"power":power,"zero_power":1/normalization,
            "multiplier":multiplier,"zero_multiplier":np.exp(-gamma*delta)}


def _spectral(coords,n_times,seed,*,max_frequency=8,gamma=-math.log(.995),kappa=2e-4,
              velocity=(.006,.003),delta=1.,sigma_e=.15,break_index=None,new_velocity=(-.003,.008)):
    if sigma_e < 0 or (break_index is not None and not 1 <= break_index < n_times):
        raise ValueError("invalid measurement sd or velocity-change row")
    spec=spectral_spec(max_frequency,gamma=gamma,kappa=kappa,velocity=velocity,delta=delta)
    changed=spectral_spec(max_frequency,gamma=gamma,kappa=kappa,velocity=new_velocity,delta=delta)
    nmode=len(spec["power"])
    initial=_rng(seed,"spectral-initial")
    state=(initial.normal(size=nmode)+1j*initial.normal(size=nmode))/np.sqrt(2)*np.sqrt(spec["power"])
    zero=initial.normal()*np.sqrt(spec["zero_power"])
    innovations=_rng(seed,"spectral-process").normal(size=(n_times,nmode,2))
    innovations=(innovations[:,:,0]+1j*innovations[:,:,1])/np.sqrt(2)
    zero_noise=_rng(seed,"spectral-zero-process").normal(size=n_times)
    scale=np.sqrt(spec["power"]*(1-np.abs(spec["multiplier"])**2))
    zero_scale=np.sqrt(spec["zero_power"]*(1-spec["zero_multiplier"]**2))
    coefficient=np.empty((n_times,nmode),complex);constant=np.empty(n_times)
    coefficient[0]=state;constant[0]=zero
    for t in range(1,n_times):
        transition=changed["multiplier"] if break_index is not None and t>=break_index else spec["multiplier"]
        state=transition*state+scale*innovations[t]
        zero=spec["zero_multiplier"]*zero+zero_scale*zero_noise[t]
        coefficient[t]=state;constant[t]=zero
    basis=np.exp(2j*np.pi*spec["wavevectors"]@coords.T)
    latent=constant[:,None]+2*np.real(coefficient@basis)
    observation=latent+sigma_e*_rng(seed,"measurement").normal(size=latent.shape)
    metadata={"gamma":gamma,"kappa":kappa,"velocity":list(velocity),"delta":delta,"sigma_e":sigma_e,
              "max_frequency":max_frequency,"full_mode_count":2*nmode+1,"stationary_point_variance":float(spec["zero_power"]+2*np.sum(spec["power"])),
              "stationary_initialization":True,"spectral_representation":"finite conjugate-paired complex coefficients; CN(0,1) half-plane innovations",
              "break_index":break_index,"new_velocity":list(new_velocity) if break_index is not None else None}
    return observation,latent,None,None,metadata


def fhn_current(physical_coordinates):
    s=np.asarray(physical_coordinates,float)
    return (.2+.32*np.exp(-np.sum((s-[9,12])**2,axis=-1)/18)
            +.28*np.exp(-np.sum((s-[23,20])**2,axis=-1)/18))


def fhn_equilibrium(current, *, a=.7,b=.8):
    """Unique equilibrium for the benchmark's monotone cubic (0<b<=1)."""
    if not 0 < b <= 1:
        raise ValueError("benchmark equilibrium constructor requires 0<b<=1")
    p=3*(1/b-1);q=3*(a/b-np.asarray(current))
    discriminant=np.sqrt(q*q/4+p**3/27)
    u=np.cbrt(-q/2+discriminant)+np.cbrt(-q/2-discriminant)
    return u,(u+a)/b


def _fhn_initial(n,seed,a,b,length):
    axis=(np.arange(n)+.5)*length/n
    x,y=np.meshgrid(axis,axis,indexing="xy")
    current=fhn_current(np.stack([x,y],axis=-1))
    u,v=fhn_equilibrium(current,a=a,b=b)
    rng=_rng(seed,"fhn-initial")
    # Fixed continuous modes make initial fields comparable across solver grids.
    basis=np.stack([np.cos(np.pi*k*x/length)*np.cos(np.pi*l*y/length)
                    for k in range(1,5) for l in range(1,5)])
    u=u+.05*np.einsum("k,kij->ij",rng.normal(size=16)/4,basis)
    v=v+.05*np.einsum("k,kij->ij",rng.normal(size=16)/4,basis)
    center=rng.uniform(8,24,2)
    u+=2.5*np.exp(-((x-center[0])**2+(y-center[1])**2)/(2*1.5**2))
    return u,v,current,center


def _fhn_reaction(u,v,current,dt,a,b,epsilon):
    def rates(u,v):
        return u-u*u*u/3-v+current,epsilon*(u+a-b*v)
    k1,l1=rates(u,v)
    k2,l2=rates(u+dt*k1/2,v+dt*l1/2)
    k3,l3=rates(u+dt*k2/2,v+dt*l2/2)
    k4,l4=rates(u+dt*k3,v+dt*l3)
    return u+dt*(k1+2*k2+2*k3+k4)/6,v+dt*(l1+2*l2+2*l3+l4)/6


def _fhn(coords,n_times,seed,*,grid_size=128,dt=.05,delta=1.,burnin=(200.,400.),
         kappa=.2,a=.7,b=.8,epsilon=.08,sigma_e=.1,domain_length=32.):
    if (not isinstance(grid_size,(int,np.integer)) or grid_size<8 or dt<=0 or delta<=0 or
            kappa<0 or epsilon<=0 or sigma_e<0 or domain_length!=32.):
        raise ValueError("invalid FHN parameters; this pacemaker benchmark uses physical domain[0,32]^2")
    steps=int(round(delta/dt))
    if not np.isclose(steps*dt,delta,rtol=0,atol=1e-10) or steps<1:
        raise ValueError("FHN observation spacing must be an integer multiple of dt")
    if np.ndim(burnin)==0:
        duration=float(burnin)
    else:
        low,high=map(float,burnin)
        if not 0<=low<=high:
            raise ValueError("burnin interval must be nonnegative and ordered")
        duration=float(_rng(seed,"fhn-burnin").uniform(low,high))
    if duration<0:
        raise ValueError("burnin must be nonnegative")
    burn_steps=int(round(duration/dt))
    u,v,current,center=_fhn_initial(grid_size,seed,a,b,domain_length)
    eigen=-4/(domain_length/grid_size)**2*np.sin(np.pi*np.arange(grid_size)/(2*grid_size))**2
    diffusion=np.exp(kappa*dt/2*(eigen[:,None]+eigen[None,:]))
    positions=np.clip(np.vstack([coords[:,1],coords[:,0]])*grid_size-.5,0,grid_size-1)
    latent=np.empty((n_times,len(coords)));recovery=np.empty_like(latent)
    max_abs=0.;energies=[]
    total=burn_steps+(n_times-1)*steps
    for iteration in range(total+1):
        if iteration>=burn_steps and (iteration-burn_steps)%steps==0:
            t=(iteration-burn_steps)//steps
            latent[t]=map_coordinates(u,positions,order=1,mode="nearest",prefilter=False)
            recovery[t]=map_coordinates(v,positions,order=1,mode="nearest",prefilter=False)
            energy=.5*domain_length**2*np.mean(u*u+v*v/epsilon)
            if not np.isfinite(energy):
                raise FloatingPointError("FHN numerical trajectory became nonfinite")
            energies.append(float(energy));max_abs=max(max_abs,float(np.max(np.abs(u))),float(np.max(np.abs(v))))
        if iteration==total:
            break
        u=idctn(diffusion*dctn(u,type=2,norm="ortho"),type=2,norm="ortho")
        u,v=_fhn_reaction(u,v,current,dt,a,b,epsilon)
        u=idctn(diffusion*dctn(u,type=2,norm="ortho"),type=2,norm="ortho")
    observation=latent+sigma_e*_rng(seed,"measurement").normal(size=latent.shape)
    metadata={"physical_domain":[0,domain_length,0,domain_length],"coordinate_transform":"physical=32*normalized",
              "grid_size":grid_size,"dt":dt,"delta":delta,"burnin_requested":list(burnin) if np.ndim(burnin) else duration,
              "burnin_realized":burn_steps*dt,"kappa":kappa,"a":a,"b":b,"epsilon":epsilon,"sigma_e":sigma_e,
              "solver":"Strang split: exact semidiscrete Neumann diffusion / RK4 reaction",
              "dgp_target":"finite cell-centered semidiscrete FHN field on the declared solver grid",
              "continuum_fidelity_claim":False,
              "observation_operator":"bilinear interpolation of cell-centered fields with no-flux boundary extension",
              "initial_patch_center":center.tolist(),"hidden_state":"recovery_v","process_innovation":None,
              "max_abs_state_at_observation_times":max_abs,"energy_min":min(energies),"energy_max":max(energies),
              "mean_temporal_sd":float(np.mean(np.std(latent,axis=0))),"numerical_refinement":"validate separately; not inferred from finite states"}
    return observation,latent,None,recovery,metadata


@lru_cache(maxsize=8)
def volatility_reference(scale_c=1.):
    """Independent fixed design reference; never uses final outcomes or sites."""
    if scale_c<=0:
        raise ValueError("volatility scale must be positive")
    phases=phase_path(4096,(.43,1.37),burnin=500)
    mean=structured_mean(phases,grid_coordinates(25))
    return float(np.sqrt(np.mean(1+.75*np.tanh(mean/scale_c))))


def _fronts(coords,n_times,seed,*,arm=None,nu=5.,scale_c=1.,pilot_rms=None,burnin=500):
    if nu<=2:
        raise ValueError("variance-matched Student noise requires nu>2")
    base=simulate_dataset(n_times,coordinates=coords,scenario="matched_gaussian",seed=seed,
                          stage="benchmark-fronts-v2",replication=0,input_sizes=(1,),burnin=burnin)
    spatial=base.latent-base.generating_mean
    measurement=base.observation-base.latent
    if arm=="whole_field_t":
        w=_rng(seed,"whole-field-mixture").chisquare(nu,size=(n_times,1))
        spatial=spatial*np.sqrt((nu-2)/w)
    elif arm=="local_t":
        w=_rng(seed,"local-mixture").chisquare(nu,size=measurement.shape)
        measurement=measurement*np.sqrt((nu-2)/w)
    elif arm=="latent_state_heteroskedastic":
        pilot_rms=volatility_reference(scale_c) if pilot_rms is None else float(pilot_rms)
        if scale_c<=0 or not np.isfinite(pilot_rms) or pilot_rms<=0:
            raise ValueError("volatility scales must be finite positive constants from an independent pilot")
        initial=np.asarray(base.parameters["initial_phase"])
        before=phase_path(1,initial,burnin=burnin-1)[0] if burnin else initial
        lag_mean=structured_mean(np.vstack([before,base.phases[:-1]]),coords)
        multiplier=np.sqrt(1+.75*np.tanh(lag_mean/scale_c))/pilot_rms
        measurement=measurement*multiplier
    elif arm is not None:
        raise ValueError("noise arm must be whole_field_t, local_t or latent_state_heteroskedastic")
    latent=base.generating_mean+spatial
    metadata={**base.parameters,"noise_arm":arm,"student_t_df":nu if arm in {"whole_field_t","local_t"} else None,
              "student_variance_matching":arm in {"whole_field_t","local_t"},
              "volatility_scale_c":scale_c if arm=="latent_state_heteroskedastic" else None,
              "volatility_pilot_rms":pilot_rms if arm=="latent_state_heteroskedastic" else None,
              "volatility_reference":"fixed4096phase steps on25x25grid; initial(.43,1.37); burnin500" if arm=="latent_state_heteroskedastic" else None,
              "legacy_source":"simulation.phase_path/structured_mean/simulate_dataset",
              "stress_component":"spatial_process_only" if arm=="whole_field_t" else "measurement_only" if arm else None}
    return latent+measurement,latent,base.generating_mean,None,metadata


def generate(scenario,n_times,*,n_side=25,n_observed=100,n_calibration=20,seed=20261102,
             coordinates=None,geometry_seed=20261102,master_observed=None,**kwargs):
    """Generate a shared dataset, then expose only declared observation views.

    G6 requires ``origin_index`` and uses the new velocity in the transition
    producing row ``origin_index + break_lead - 1`` (default lead25).
    Horizon extension preserves the common prefix for a fixed DGP/domain.
    """
    name=ALIASES.get(scenario,scenario)
    if name not in SCENARIOS:
        raise ValueError(f"unknown DGP {scenario!r}; choose from {SCENARIOS}")
    if not isinstance(n_times,(int,np.integer)) or n_times<2 or not isinstance(seed,(int,np.integer)) or seed<0:
        raise ValueError("n_times must be integer>=2 and seed a nonnegative integer")
    periodic=name in {"g2_spectral_transport","g6_velocity_shift"}
    coords=grid_coordinates(n_side,periodic=periodic) if coordinates is None else _coordinates(coordinates)
    if periodic and np.any(coords>=1):
        raise ValueError("periodic coordinates must lie in [0,1), without duplicate endpoints")
    design=make_design(coords,n_observed,n_calibration,seed=geometry_seed,master_observed=master_observed)
    if name=="g1_gaussian_ar":
        output=_gaussian_ar(coords,n_times,seed,**kwargs)
    elif name in {"g2_spectral_transport","g6_velocity_shift"}:
        if name=="g6_velocity_shift":
            if "origin_index" not in kwargs:
                raise ValueError("G6 requires an explicit forecast origin_index")
            origin=kwargs.pop("origin_index");lead=kwargs.pop("break_lead",25)
            if not isinstance(origin,(int,np.integer)) or not isinstance(lead,(int,np.integer)) or origin<1 or lead<1:
                raise ValueError("origin_index and break_lead must be positive integers")
            kwargs["break_index"]=int(origin+lead-1)
        output=_spectral(coords,n_times,seed,**kwargs)
    elif name=="g3_excitable_field":
        output=_fhn(coords,n_times,seed,**kwargs)
    else:
        if name=="g5_noise_stress" and "arm" not in kwargs:
            kwargs["arm"]="local_t"
        output=_fronts(coords,n_times,seed,**kwargs)
    observed,latent,mean,hidden,parameters=output
    delta=float(parameters.get("delta",1.))
    metadata={"schema_version":"benchmark-dgp-2.0","scenario":name,"seed":int(seed),
              "geometry_seed":int(geometry_seed),"n_times":int(n_times),"n_sites":len(coords),
              "n_observed":len(design.observed_indices),"n_calibration":len(design.calibration_indices),
              "n_test":len(design.test_indices),"master_observed_indices":design.master_observed_indices.tolist(),
              "inactive_sensor_indices":design.inactive_sensor_indices.tolist(),"hole":list(HOLE),
              "boundary":"periodic" if periodic else "no_flux" if name=="g3_excitable_field" else "nonperiodic",
              "parameters":parameters,"numpy_version":np.__version__,
              "observation_sha256":hashlib.sha256(np.ascontiguousarray(observed,dtype="<f8").tobytes()).hexdigest(),
              "geometry_sha256":hashlib.sha256(np.ascontiguousarray(coords,dtype="<f8").tobytes()).hexdigest()}
    return Simulation(observed,latent,coords,np.arange(n_times)*delta,
                      design.observed_indices,design.calibration_indices,design.test_indices,
                      design.hole_indices,metadata,mean,hidden)


def gaussian_ar_oracle_moments(y_observed,observed_coordinates,target_coordinates,horizon,
                               *,rho=.97,ell=.2,sigma_e=.15):
    """Exact known-parameter forecast moments using noisy O history only.

    Never accepts true latent states or U observations. All spatial inverse
    notation is implemented by solves. Filtering is in K_OO eigenmodes;
    unobserved spatial residuals remain independent AR processes.
    """
    y=np.asarray(y_observed,float)
    obs=_coordinates(observed_coordinates);target=_coordinates(target_coordinates)
    if y.ndim!=2 or len(y)<1 or y.shape[1]!=len(obs) or not np.isfinite(y).all() or horizon<1 or not abs(rho)<1 or sigma_e<0:
        raise ValueError("invalid oracle data or parameters")
    koo=matern32(obs,ell=ell);kto=matern32(target,obs,ell=ell);ktt=matern32(target,ell=ell)
    values,vectors=eigh(koo)
    transformed=y@vectors
    mean=np.zeros(len(obs));variance=values.copy()
    for t,row in enumerate(transformed):
        if t:
            mean*=rho;variance=rho*rho*variance+(1-rho*rho)*values
        gain=variance/(variance+sigma_e**2)
        mean+=gain*(row-mean);variance*=1-gain
    weights=cho_solve(cho_factor(koo,lower=True),kto.T).T
    terminal_mean=weights@(vectors@mean)
    terminal_cov=weights@((vectors*variance)@vectors.T)@weights.T+ktt-weights@kto.T
    terminal_cov=(terminal_cov+terminal_cov.T)/2
    powers=rho**np.arange(1,horizon+1)
    future_mean=powers[:,None]*terminal_mean
    future_var=powers[:,None]**2*np.diag(terminal_cov)+(1-powers[:,None]**2)*np.diag(ktt)+sigma_e**2
    return {"mean":future_mean,"variance":future_var,"terminal_latent_mean":terminal_mean,
            "terminal_latent_covariance":terminal_cov,"spatial_covariance":ktt,
            "parameters":{"rho":rho,"ell":ell,"sigma_e":sigma_e},
            "information_policy":"known parameters; observed-site noisy history only"}
