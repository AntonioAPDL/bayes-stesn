"""Coordinate-basis Gaussian dynamics with a constrained generalized EM fit.

The fixed decoder obeys Phi'Phi/n=I. Filtering uses the r-dimensional
projection with measurement variance R/n, and separately accounts for the
orthogonal observation likelihood. No n-by-n covariance is factored per time.
"""
from __future__ import annotations
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from .classical import (ClassicalFit, CoordinateBasis, _positive, _horizon, _paths,
                        psd_root)
from .contracts import Forecast, validate_history, forecast_targets
from ..io import seed_for


DEFAULT_CONFIG = {"rank":12,"basis_width":.25,"initial_variance":1.,
    "transition_norm_cap":.995,"variance_floor":1e-6,"max_iter":40,
    "tolerance":1e-5,"mstep_steps":100,"max_forecast_bytes":1_073_741_824}


def reduced_filter(b, A, q, noise, n_sites, *, initial_mean=None,
                   initial_cov=None, predict_initial=False, store=False):
    """Filter projected observations b=Y Phi/n; returns their log density.

    To obtain the full observation density, subtract T*r*log(n)/2 and add
    the N(0,noise) density in the n-r orthogonal directions.
    """
    b = np.asarray(b, float); r = A.shape[0]
    m = np.zeros(r) if initial_mean is None else np.array(initial_mean, copy=True)
    p = np.eye(r) if initial_cov is None else np.array(initial_cov, copy=True)
    q_matrix = np.diag(q)
    measurement = noise/n_sites
    ll = 0.
    histories = None
    if store:
        histories = {"m":np.empty_like(b), "p":np.empty((len(b),r,r)),
                     "pred_m":np.empty_like(b), "pred_p":np.empty((len(b),r,r))}
    for t, y in enumerate(b):
        if t or predict_initial:
            m = A @ m
            p = A @ p @ A.T + q_matrix
        if store:
            histories["pred_m"][t] = m; histories["pred_p"][t] = p
        innovation = y-m
        factor = cho_factor(p+measurement*np.eye(r), lower=True, check_finite=False)
        gain = cho_solve(factor, p.T, check_finite=False).T
        m = m+gain @ innovation
        p = p-gain @ p
        p = (p+p.T)/2
        ll -= .5*(r*np.log(2*np.pi)+2*np.log(np.diag(factor[0])).sum()
                   +innovation @ cho_solve(factor,innovation,check_finite=False))
        if store:
            histories["m"][t] = m; histories["p"][t] = p
    return m, p, float(ll), histories


def rts_smooth(history, A):
    """RTS state moments and Cov(a[t],a[t-1] | all observations)."""
    m = history["m"].copy(); p = history["p"].copy()
    lag = np.empty((len(m)-1, A.shape[0], A.shape[0]))
    for t in range(len(m)-2, -1, -1):
        factor = cho_factor(history["pred_p"][t+1], lower=True, check_finite=False)
        gain = cho_solve(factor, A @ history["p"][t], check_finite=False).T
        m[t] += gain @ (m[t+1]-history["pred_m"][t+1])
        p[t] += gain @ (p[t+1]-history["pred_p"][t+1]) @ gain.T
        p[t] = (p[t]+p[t].T)/2
        lag[t] = p[t+1] @ gain.T
    return m, p, lag


def _spectral_projection(a, cap):
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    return (u*np.minimum(s, cap)) @ vt


def constrained_transition(s00, s10, q, old, cap, steps=100):
    """Generalized M-step minimizing a convex weighted quadratic.

    The spectral-norm ball guarantees stability. Projected gradient is part
    of the constrained M-step; there is no post-fit projection disguised as
    unconstrained EM. A step at inverse gradient Lipschitz constant cannot
    increase this objective (apart from roundoff).
    """
    unconstrained = cho_solve(cho_factor(s00, lower=True), s10.T).T
    if np.linalg.norm(unconstrained, 2) <= cap:
        return unconstrained
    a = np.array(old, copy=True)
    step = float(np.min(q))/max(float(np.linalg.eigvalsh(s00)[-1]), 1e-15)
    for _ in range(steps):
        gradient = (a @ s00-s10)/q[:, None]
        candidate = _spectral_projection(a-step*gradient, cap)
        if np.linalg.norm(candidate-a) < 1e-8*(1+np.linalg.norm(a)):
            return candidate
        a = candidate
    return a


class LinearStateFit(ClassicalFit):
    model_id = "basis_lgssm"
    _state_fields = ("center", "scale", "A", "q", "noise", "initial_variance",
                     "terminal_mean", "terminal_covariance", "transition_norm_cap")

    def __init__(self, data, config=None, seed=0):
        super().__init__(data, config, seed)
        self.basis = CoordinateBasis.fit(data.coordinates, self.config.get("rank", 12),
                                         self.config.get("basis_width", .25))
        self._phi = self.basis.evaluate(data.coordinates)
        self.center = float(data.y.mean())
        self.scale = max(float(data.y.std()), 1e-8)
        y = (data.y-self.center)/self.scale
        n = y.shape[1]; r = self._phi.shape[1]
        b = y @ self._phi/n
        orthogonal_ss = max(float(np.sum(y*y)-n*np.sum(b*b)), 0.)
        self.initial_variance = _positive(self.config.get("initial_variance", 1.), "initial_variance")
        self.transition_norm_cap = float(self.config.get("transition_norm_cap", .995))
        if not 0 < self.transition_norm_cap < 1:
            raise ValueError("transition_norm_cap must lie in (0,1)")
        initial_ridge = 1e-6*np.eye(r)
        a = np.linalg.solve(b[:-1].T @ b[:-1]+initial_ridge, b[:-1].T @ b[1:]).T
        self.A = _spectral_projection(a, self.transition_norm_cap)
        floor = _positive(self.config.get("variance_floor", 1e-6), "variance_floor")
        self.q = np.maximum(np.var(b[1:]-b[:-1] @ self.A.T, axis=0), floor)
        self.noise = max(orthogonal_ss/max(len(y)*(n-r), 1), .05, floor)
        maximum = int(self.config.get("max_iter", 40))
        if maximum < 1:
            raise ValueError("max_iter must be positive")
        tolerance = _positive(self.config.get("tolerance", 1e-5), "tolerance")
        likelihoods=[]; converged=False; rejected=False
        last_params = None
        for iteration in range(maximum+1):
            terminal_m, terminal_p, reduced_ll, history = reduced_filter(
                b, self.A, self.q, self.noise, n,
                initial_cov=self.initial_variance*np.eye(r), store=True)
            ll = reduced_ll-.5*len(y)*r*np.log(n)-.5*(len(y)*(n-r)*np.log(2*np.pi*self.noise)+orthogonal_ss/self.noise)
            # The reported likelihood is on the original response scale.
            ll -= len(y)*n*np.log(self.scale)
            if not np.isfinite(ll):
                raise FloatingPointError("LGSSM fit has a nonfinite likelihood")
            if likelihoods and ll < likelihoods[-1]-1e-7*(1+abs(likelihoods[-1])):
                self.A, self.q, self.noise = last_params
                rejected=True
                break
            likelihoods.append(float(ll))
            self.terminal_mean, self.terminal_covariance = terminal_m, terminal_p
            if len(likelihoods)>1 and ll-likelihoods[-2] <= tolerance*(1+abs(likelihoods[-2])):
                converged=True
                break
            if iteration == maximum:
                break
            ms, ps, lag = rts_smooth(history, self.A)
            s00 = ms[:-1].T @ ms[:-1] + ps[:-1].sum(axis=0)
            s11 = ms[1:].T @ ms[1:] + ps[1:].sum(axis=0)
            s10 = ms[1:].T @ ms[:-1] + lag.sum(axis=0)
            new_a = constrained_transition(s00, s10, self.q, self.A,
                                           self.transition_norm_cap,
                                           steps=int(self.config.get("mstep_steps", 100)))
            residual = s11-new_a @ s10.T-s10 @ new_a.T+new_a @ s00 @ new_a.T
            new_q = np.maximum(np.diag(residual)/(len(y)-1), floor)
            error = y-ms @ self._phi.T
            new_noise = max(float((np.sum(error*error)+n*np.trace(ps.sum(axis=0)))/(len(y)*n)), floor)
            last_params = (self.A.copy(), self.q.copy(), self.noise)
            self.A, self.q, self.noise = new_a, new_q, new_noise
        self.diagnostics.update(converged=converged, qualified=bool(converged and not rejected), finite_fit=True,
            inference="constrained_generalized_EM", rank=r, iterations=len(likelihoods)-1,
            log_likelihood=likelihoods, rejected_decreasing_step=rejected,
            transition_spectral_radius=float(max(abs(np.linalg.eigvals(self.A)))),
            transition_spectral_norm=float(np.linalg.norm(self.A,2)),
            noise_variance_original=float(self.noise*self.scale**2),
            variance_floor_hits=int(np.sum(self.q<=floor*(1+1e-8)))+int(self.noise<=floor*(1+1e-8)),
            uncertainty="Gaussian_state_and_observation_uncertainty_given_fitted_parameters",
            initial_state="fixed_proper_N(0,initial_variance_I)",
            mean="global_training_mean_evaluable_at_every_site",
            optimizer_budget={"max_iter":maximum,"tolerance":tolerance,
                              "mstep_steps":int(self.config.get("mstep_steps",100))},
            stop_reason="decreasing_step_rejected" if rejected else "relative_likelihood_tolerance" if converged else "iteration_cap",
            final_relative_likelihood_gain=(None if len(likelihoods)<2 else
                (likelihoods[-1]-likelihoods[-2])/(1+abs(likelihoods[-2]))))

    def _after_load(self):
        self._phi = self.basis.evaluate(self.data.coordinates)

    def _terminal(self, observed):
        if len(observed.y) == len(self.data.y):
            return self.terminal_mean, self.terminal_covariance
        extra = (observed.y[len(self.data.y):]-self.center)/self.scale
        b = extra @ self._phi/len(self._phi)
        m,p,_,_ = reduced_filter(b,self.A,self.q,self.noise,len(self._phi),
            initial_mean=self.terminal_mean,initial_cov=self.terminal_covariance,predict_initial=True)
        return m,p

    def forecast(self, horizon, target_coordinates=None, target_site_ids=None,
                 n_paths=256, seed=0, history=None):
        h = _horizon(horizon); paths = _paths(n_paths)
        observed = validate_history(self.data, history)
        coords, ids = forecast_targets(self.data, target_coordinates, target_site_ids)
        unique, mapping = np.unique(coords, axis=0, return_inverse=True)
        required=8*(2*paths*h*len(coords)+4*h*len(coords)
                    +4*paths*(self._phi.shape[1]+len(unique)))
        if required > int(self.config.get("max_forecast_bytes", 1_073_741_824)):
            raise MemoryError("Forecast allocation exceeds max_forecast_bytes")
        phi = self.basis.evaluate(unique)
        m,p = self._terminal(observed)
        mean=np.empty((h,len(unique))); current=m.copy()
        for lead in range(h):
            current = self.A @ current
            mean[lead] = self.center+self.scale*(phi @ current)
        samples=None
        if paths:
            r=len(m); streams=[np.random.default_rng(seed_for(seed,self.model_id,"path",i)) for i in range(paths)]
            initial=np.array([rng.normal(size=r) for rng in streams])
            states=m+initial @ psd_root(p).T
            samples=np.empty((paths,h,len(unique)))
            for lead in range(h):
                noise=np.array([rng.normal(size=r+len(unique)) for rng in streams])
                states=states @ self.A.T+noise[:,:r]*np.sqrt(self.q)
                samples[:,lead] = self.center+self.scale*(states @ phi.T+np.sqrt(self.noise)*noise[:,r:])
            samples=samples[:,:,mapping]
        return Forecast(mean=mean[:,mapping],samples=samples,coordinates=coords,site_ids=ids,
                        metadata=self._metadata(observed,uncertainty="joint_plugin_gaussian",
                                                parameter_uncertainty=False,
                                                rank=len(m),path_seed_policy="global_path_index"))
