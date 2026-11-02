"""Independent dense, distributional, transform, and exact-resumption checks."""

import copy
import json
from dataclasses import FrozenInstanceError, dataclass, replace
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg, special, stats

from bayes_stesn.gaussian import GaussianRegression, PrecisionSpec
from bayes_stesn.priors import (PriorRuntime, PriorSpec, half_cauchy_log_coordinate,
                               inverse_gamma_log_coordinate, residualize_correction,
                               rhs_log_variance, sample_gig)
from bayes_stesn.sampling import (RegressionPosterior, SamplerConfig,
                                  _calibrate_initialization, sample_regression)
from bayes_stesn.spatial import CovarianceSpec


def dense_reference(Y,Phi,Z,R,precision,a0=2.,b0=.05,rho=0.,sigma2=None):
    """Integrate using the observation-space covariance, not spectral identities."""
    n,T = Y.shape
    X = np.kron(Z,Phi)
    y = Y.reshape(-1,order="F")
    S = rho**np.abs(np.arange(T)[:,None]-np.arange(T)[None,:])
    Omega0 = np.kron(S,R)
    Lambda = precision.as_matrix()
    Omega = Omega0+X@linalg.solve(Lambda,X.T,assume_a="pos")
    LO = linalg.cholesky(Omega,lower=True)
    Q = y@linalg.cho_solve((LO,True),y)
    logdet = 2*np.log(np.diag(LO)).sum()
    N = n*T
    if sigma2 is None:
        an,bn = a0+N/2,b0+Q/2
        loglik = -.5*N*np.log(2*np.pi)-.5*logdet+a0*np.log(b0)+special.gammaln(an)-special.gammaln(a0)-an*np.log(bn)
    else:
        loglik = -.5*N*np.log(2*np.pi*sigma2)-.5*logdet-.5*Q/sigma2
    RiX = linalg.solve(Omega0,X,assume_a="pos")
    A = Lambda+X.T@RiX
    m = linalg.solve(A,RiX.T@y,assume_a="pos")
    return loglik,Q,m,linalg.solve(A,np.eye(len(m)),assume_a="pos")


class GaussianAlgebraTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(410)
        self.Phi = rng.normal(size=(4,2))
        self.Z = rng.normal(size=(7,3))
        self.Y = rng.normal(size=(4,7))
        M = rng.normal(size=(4,4))
        self.R = np.eye(4)+.2*M@M.T
        self.engine = GaussianRegression(self.Y,self.Phi,self.Z)

    def test_scalar_column_diagonal_dense_match_observation_space(self):
        rng = np.random.default_rng(91)
        U = rng.normal(size=(6,6))
        priors = [PrecisionSpec(2,3,"scalar",.7),
                  PrecisionSpec(2,3,"column",[.3,1.2,3.]),
                  PrecisionSpec(2,3,"diagonal",[.3,.6,1.,1.2,2.,3.]),
                  PrecisionSpec(2,3,"dense",np.eye(6)+U@U.T)]
        for precision in priors:
            with self.subTest(kind=precision.kind):
                ev = self.engine.evaluate(self.R,precision=precision)
                loglik,Q,mean,_ = dense_reference(self.Y,self.Phi,self.Z,self.R,precision)
                assert_allclose(ev.log_marginal,loglik,rtol=2e-12,atol=2e-12)
                assert_allclose(ev.Q,Q,rtol=2e-12,atol=2e-12)
                assert_allclose(ev.mean.reshape(-1,order="F"),mean,rtol=2e-11,atol=2e-12)
                fixed = self.engine.evaluate(self.R,precision=precision,sigma2=.4)
                reference = dense_reference(self.Y,self.Phi,self.Z,self.R,precision,sigma2=.4)[0]
                assert_allclose(fixed.log_marginal,reference,rtol=2e-12,atol=2e-12)

    def test_rank_deficient_design_remains_proper(self):
        Z = np.column_stack((self.Z[:,0],self.Z[:,0],np.zeros(7)))
        Phi = np.column_stack((self.Phi[:,0],self.Phi[:,0]))
        engine = GaussianRegression(self.Y,Phi,Z)
        prec = PrecisionSpec(2,3,"column",[.2,.7,1.3])
        ev = engine.evaluate(self.R,precision=prec)
        expected = dense_reference(self.Y,Phi,Z,self.R,prec)
        assert_allclose(ev.log_marginal,expected[0],atol=2e-11)
        assert_allclose(ev.mean.reshape(-1,order="F"),expected[2],atol=2e-11)

    def test_ar_likelihood_matches_full_toeplitz_model(self):
        prec = PrecisionSpec(2,3,"column",[.3,1.1,2.])
        for rho in (-.65,0.,.78):
            ev = self.engine.evaluate(self.R,precision=prec,temporal_rho=rho)
            expected = dense_reference(self.Y,self.Phi,self.Z,self.R,prec,rho=rho)
            assert_allclose(ev.log_marginal,expected[0],atol=1e-10)
            assert_allclose(ev.mean.reshape(-1,order="F"),expected[2],atol=2e-11)
            Yt,Zt = self.Y.copy(),self.Z.copy()
            Yt[:,1:] = (self.Y[:,1:]-rho*self.Y[:,:-1])/np.sqrt(1-rho*rho)
            Zt[1:] = (self.Z[1:]-rho*self.Z[:-1])/np.sqrt(1-rho*rho)
            G,YY,YZ,logdet = self.engine.statistics(rho)
            assert_allclose(G,Zt.T@Zt,atol=2e-12)
            assert_allclose(YY,Yt@Yt.T,atol=2e-12)
            assert_allclose(YZ,Yt@Zt,atol=2e-12)
            assert_allclose(logdet,6*np.log1p(-rho*rho))

    def test_joint_recovery_mean_covariance_and_scale(self):
        rng = np.random.default_rng(991)
        prec = PrecisionSpec(2,3,"column",[.6,1.1,2.])
        ev = self.engine.evaluate(self.R,precision=prec)
        _,_,mean,cov = dense_reference(self.Y,self.Phi,self.Z,self.R,prec)
        samples = [ev.draw(rng) for _ in range(7000)]
        b = np.array([x[0].reshape(-1,order="F") for x in samples])
        scales = np.array([x[1] for x in samples])
        expected_cov = ev.bn/(ev.an-1)*cov
        standardized_mean_error = (b.mean(axis=0)-mean)/np.sqrt(np.diag(expected_cov)/len(b))
        self.assertLess(abs(standardized_mean_error).max(),4.5)
        self.assertLess(linalg.norm(np.cov(b.T)-expected_cov)/linalg.norm(expected_cov),.065)
        assert_allclose(scales.mean(),ev.bn/(ev.an-1),rtol=.02)

    def test_sigma2_conditional_counts_coefficient_normalizers(self):
        rng = np.random.default_rng(17)
        C = rng.normal(size=(2,3))
        prec = PrecisionSpec(2,3,"column",[.4,1.,2.])
        residual = self.Y-self.Phi@C@self.Z.T
        scale = .05+.5*(np.sum(residual*linalg.solve(self.R,residual,assume_a="pos"))+
                        C.reshape(-1,order="F")@prec.as_matrix()@C.reshape(-1,order="F"))
        actual = self.engine.draw_sigma2_conditional(C,self.R,prec,np.random.default_rng(89))
        expected = scale/np.random.default_rng(89).gamma(2+(28+6)/2)
        assert_allclose(actual,expected,rtol=2e-15)

    def test_matrixfree_draws_and_failure_policy(self):
        prec = PrecisionSpec(2,3,"diagonal",[.25,.6,1.,2.,3.,4.])
        exact = self.engine.evaluate(self.R,precision=prec,sigma2=.6)
        _,_,mean,cov = dense_reference(self.Y,self.Phi,self.Z,self.R,prec,sigma2=.6)
        rng = np.random.default_rng(287)
        draws = []
        for _ in range(4000):
            C,meta = self.engine.draw_conditional(self.R,prec,.6,rng,"matrixfree",rtol=1e-12)
            self.assertTrue(meta["converged"])
            self.assertTrue(meta["finite_tolerance_approximation"])
            self.assertLess(meta["relative_residual"],1.1e-12)
            draws.append(C.reshape(-1,order="F"))
        draws = np.array(draws)
        self.assertLess(linalg.norm(np.cov(draws.T)-.6*cov)/linalg.norm(.6*cov),.08)
        self.assertLess(abs((draws.mean(0)-mean)/np.sqrt(.6*np.diag(cov)/len(draws))).max(),4.5)
        good,_ = self.engine.draw_conditional(self.R,prec,.6,np.random.default_rng(11),"matrixfree",rtol=1e-13)
        fallback,meta = self.engine.draw_conditional(self.R,prec,.6,np.random.default_rng(11),"matrixfree",rtol=1e-13,maxiter=1)
        self.assertTrue(meta["dense_fallback"])
        assert_allclose(fallback,good,atol=2e-12)
        limited = GaussianRegression(self.Y,self.Phi,self.Z,dense_threshold=1)
        with self.assertRaises(linalg.LinAlgError):
            limited.draw_conditional(self.R,prec,.6,np.random.default_rng(11),"matrixfree",maxiter=1)
        with self.assertRaises(MemoryError):
            limited.evaluate(self.R,precision=prec)

    def test_cancellation_safe_quadratic_and_chunks(self):
        rng = np.random.default_rng(91)
        Z = rng.normal(size=(37,3))
        C = rng.normal(size=(2,3))*1000
        Y = self.Phi@C@Z.T+rng.normal(size=(4,37))*1e-5
        engine = GaussianRegression(Y,self.Phi,Z)
        for rho in (0.,.7):
            prec = PrecisionSpec(2,3,"scalar",1e-12)
            ev = engine.evaluate(self.R,precision=prec,temporal_rho=rho)
            stable = engine.residual_quadratic(ev.mean,self.R,rho,chunk_size=5)+prec.quadratic(ev.mean)
            self.assertGreater(ev.Q,0)
            assert_allclose(ev.Q,stable,rtol=2e-7,atol=1e-14)


class PriorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        self.Phi = np.c_[np.ones(5),rng.normal(size=5)]
        self.Z = np.c_[np.ones(12),rng.normal(size=(12,3))]
        self.groups = ("intercept","score","reservoir_1","reservoir_1")
        self.R = np.eye(5)*1.7

    def runtime(self,spec):
        return PriorRuntime(spec,2,4,self.groups,self.Phi,self.Z)

    def test_rhs_density_includes_correct_jacobians(self):
        runtime = self.runtime(PriorSpec.rhs(tau0=.2,slab_scale=1.3,slab_df=3.))
        state = runtime.initial_state()
        state.update(log_local=np.array([-.8,.3]),log_tau=-1.2,log_c2=.5)
        u,t,z = state["log_local"],state["log_tau"],state["log_c2"]
        expected = np.sum(stats.halfcauchy.logpdf(np.exp(u))+u)
        expected += stats.halfcauchy.logpdf(np.exp(t),scale=.2)+t
        expected += stats.invgamma.logpdf(np.exp(z),a=1.5,scale=1.5*1.3**2)+z
        assert_allclose(runtime.log_prior(state),expected,atol=1e-12)
        natural = np.exp(z)*np.exp(2*t+2*u)/(np.exp(z)+np.exp(2*t+2*u))
        assert_allclose(np.exp(rhs_log_variance(u,t,z)),natural,rtol=2e-15)
        extreme = rhs_log_variance(np.array([-500.,500.]),-10.,.5)
        self.assertTrue(np.isfinite(extreme).all())
        self.assertTrue(np.all(extreme<=.5+1e-12))
        precision = runtime.precision(state)
        assert_allclose(precision.column_values[0],.1)

    def test_reference_energy_and_variance_budget(self):
        runtime = self.runtime(PriorSpec.reference_r2(alpha=(.7,1.8)))
        state = {"logit_r":.4,"allocation_logits":np.array([-.5])}
        X = np.kron(self.Z,self.Phi)
        Xc = X-X.mean(axis=0)
        energies = []
        for group in runtime.selected_group_names:
            mask = np.repeat(np.array(self.groups)==group,2)
            energies.append(np.square(Xc[:,mask]).sum()/len(X))
        assert_allclose(runtime.reference_energy,energies,rtol=2e-15)
        precision = runtime.precision(state,self.R)
        allocated_variance = sum(e/d for e,d in zip(energies,
            [precision.column_values[np.where(np.array(self.groups)==g)[0][0]] for g in runtime.selected_group_names]))
        W = 1.7*np.exp(.4)
        assert_allclose(allocated_variance,W,rtol=2e-15)
        assert_allclose(allocated_variance/(allocated_variance+1.7),special.expit(.4))

    def test_rhs_tradeoff_preserves_likelihood_and_uses_only_hyperprior_ratio(self):
        C = np.random.default_rng(93).normal(size=(2,4))
        for unit in ("group","column","entry"):
            runtime = self.runtime(PriorSpec.rhs(unit=unit))
            state = runtime.initial_state()
            state["log_local"] = np.linspace(-.6,.7,runtime.nunits)
            proposed = copy.deepcopy(state)
            proposed["log_tau"] += .83
            proposed["log_local"] -= .83
            assert_allclose(runtime.precision(state).as_matrix(),runtime.precision(proposed).as_matrix(),rtol=2e-15)
            conditional_difference = runtime.conditional_log_target(C,.7,proposed)-runtime.conditional_log_target(C,.7,state)
            hyper_difference = runtime.log_prior(proposed)-runtime.log_prior(state)
            assert_allclose(conditional_difference,hyper_difference,rtol=2e-10,atol=3e-12)
            rng_expected = np.random.default_rng(312)
            delta = rng_expected.normal(scale=.5)
            candidate = copy.deepcopy(state)
            candidate["log_tau"] += delta
            candidate["log_local"] -= delta
            expected_accept = np.log(rng_expected.uniform())<runtime.log_prior(candidate)-runtime.log_prior(state)
            actual,accepted = runtime.rhs_tradeoff(state,np.random.default_rng(312),.5)
            self.assertEqual(accepted,expected_accept)
            assert_array_equal(runtime.vector(actual),runtime.vector(candidate if expected_accept else state))
            rng = np.random.default_rng(81)
            before = copy.deepcopy(rng.bit_generator.state)
            unchanged,accepted = runtime.rhs_tradeoff(state,rng,0.)
            assert_array_equal(runtime.vector(unchanged),runtime.vector(state))
            self.assertFalse(accepted)
            self.assertEqual(before,rng.bit_generator.state)

    def test_allocation_coordinate_densities(self):
        runtime = self.runtime(PriorSpec.reference_r2(alpha=(.7,1.8),beta_a=1.2,beta_b=2.3))
        state = {"logit_r":.4,"allocation_logits":np.array([-.5])}
        r = special.expit(.4)
        pi = special.softmax([-.5,0.])
        beta = stats.beta.logpdf(r,1.2,2.3)+np.log(r)+np.log1p(-r)
        expected = beta+stats.dirichlet.logpdf(pi,[.7,1.8])+np.log(pi).sum()
        assert_allclose(runtime.log_prior(state),expected,atol=2e-15)
        logistic = self.runtime(PriorSpec.reference_r2(allocation="logistic_normal",beta_a=1.2,beta_b=2.3,
                                                      logistic_mean=(.2,),logistic_cov=((.7,),)))
        expected = beta+stats.norm.logpdf(-.5,.2,np.sqrt(.7))
        assert_allclose(logistic.log_prior(state),expected,atol=2e-15)

    def test_gig_parameterization_negative_order_and_reciprocal(self):
        order,alpha,chi = -2.3,2.,1.8
        rng = np.random.default_rng(419)
        draws = sample_gig(order,alpha,chi,rng,size=18000)
        z = np.sqrt(alpha*chi)
        mean = np.sqrt(chi/alpha)*special.kve(order+1,z)/special.kve(order,z)
        inverse_mean = np.sqrt(alpha/chi)*special.kve(order-1,z)/special.kve(order,z)
        assert_allclose(draws.mean(),mean,rtol=.025)
        assert_allclose((1/draws).mean(),inverse_mean,rtol=.025)
        reciprocal = 1/sample_gig(-order,chi,alpha,np.random.default_rng(420),size=18000)
        self.assertLess(stats.ks_2samp(draws,reciprocal).statistic,.025)
        with self.assertRaises(ValueError):
            sample_gig(-1.,2.,0.,rng)
        gamma = sample_gig(2.,3.,0.,np.random.default_rng(5),size=4)
        assert_array_equal(gamma,np.random.default_rng(5).gamma(2.,2/3,size=4))

    def test_gigg_group_conditional_matches_declared_gig_density(self):
        runtime = self.runtime(PriorSpec.gigg(shape_a=(.7,1.2),shape_b=(.6,.9)))
        state = runtime.initial_state()
        C = np.random.default_rng(7).normal(size=(2,4))
        b = C.reshape(-1,order="F")[runtime.selected]
        sigma2 = .6
        tau2 = np.exp(state["log_tau2"])
        group = 1
        mask = runtime.selected_group==group
        m = int(mask.sum())
        chi = float(np.square(b[mask]).sum()/(sigma2*tau2))
        order = runtime.ag[group]-m/2
        values = np.array([.35,1.2,3.7])
        actual = []
        for value in values:
            candidate = copy.deepcopy(state)
            candidate["log_group_variance"][group] = np.log(value)
            # Runtime's density is on log(v), so remove the log Jacobian.
            actual.append(runtime.conditional_log_target(C,sigma2,candidate)-np.log(value))
        expected = stats.geninvgauss.logpdf(values,order,np.sqrt(2*chi),scale=np.sqrt(chi/2))
        assert_allclose(np.array(actual)-actual[0],expected-expected[0],rtol=2e-12,atol=2e-12)
        updated = runtime.update_conditional(C,sigma2,state,np.random.default_rng(61))
        self.assertTrue(np.isfinite(runtime.vector(updated)).all())
        precision = runtime.precision(updated)
        assert_allclose(precision.diagonal()[:2],.1)

    def test_subspace_residualization_and_proper_penalty(self):
        transformed,mapping = residualize_correction(self.Z,self.Z,2)
        assert_allclose(transformed[:,:2].T@transformed[:,2:],np.zeros((2,2)),atol=2e-14)
        future = np.random.default_rng(22).normal(size=(8,4))
        applied,_ = residualize_correction(self.Z,future,2)
        assert_allclose(applied[:,2:],future[:,2:]-future[:,:2]@mapping)
        runtime = self.runtime(PriorSpec.function_penalty(base_columns=2,lambda0=.4))
        X = np.kron(self.Z,self.Phi)
        X0 = np.kron(self.Z[:,:2],self.Phi)
        F0 = X.T@(np.eye(len(X))-X0@np.linalg.pinv(X0))@X/len(X)
        # The independent dense projector subtracts nearly equal matrices;
        # roundoff in its nominally zero baseline block is about 1e-14.
        assert_allclose(runtime.F0,F0,atol=2e-14)
        state = {"log_tau":-.3}
        precision = runtime.precision(state)
        assert_allclose(precision.as_matrix(),.4*np.eye(8)+np.exp(.6)*F0,atol=3e-14)
        self.assertGreater(np.linalg.eigvalsh(precision.as_matrix()).min(),.39999999)
        self.assertAlmostEqual(precision.logdet,np.linalg.slogdet(precision.as_matrix())[1],places=12)
        labelled = PriorRuntime(PriorSpec.correction(),2,4,["base","base","correction","correction"],
                                self.Phi,self.Z)
        assert_allclose(labelled.precision(labelled.initial_state()).column_values,[.1,.1,100.,100.])

    def test_specs_immutable_and_invalid_energy_refused(self):
        prior = PriorSpec.ridge()
        with self.assertRaises(FrozenInstanceError):
            prior.kind="gigg"
        with self.assertRaises(ValueError):
            PriorRuntime(PriorSpec.reference_r2(),1,2,["intercept","constant"],np.ones((3,1)),np.ones((4,2)))
        with self.assertRaises(MemoryError):
            self.runtime(PriorSpec.function_penalty(base_columns=2,max_penalty_size=3))
        with self.assertRaises(ValueError):
            PriorRuntime(PriorSpec.reference_r2(allocation="logistic_normal",logistic_cov=((1.,.8),(.1,1.))),
                         2,4,["intercept","a","b","c"],self.Phi,self.Z)


@dataclass(frozen=True)
class FixedIdentityCovariance:
    """A test-only known covariance for one-dimensional quadrature validation."""
    kind: str = "matern"
    scale_mode: str = "relative"
    infer_nu: bool = False
    def parameter_names(self,coordinates=None): return ()
    def initial_parameters(self,coordinates): return {"nu":1.5}
    def parameters_to_vector(self,params,coordinates=None): return np.zeros(0)
    def vector_to_parameters(self,vector,coordinates=None,nu=None): return {"nu":1.5}
    def log_prior(self,params,coordinates=None): return 0.
    def covariance(self,coordinates,params): return np.eye(len(coordinates))


class SamplingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(312)
        self.Phi = rng.normal(size=(5,2))
        self.Z = np.c_[np.ones(15),rng.normal(size=(15,3))]
        self.Y = self.Phi@rng.normal(size=(2,4))@self.Z.T+.3*rng.normal(size=(5,15))
        self.coordinates = rng.uniform(size=(5,2))
        self.groups = ("intercept","score","reservoir_1","reservoir_1")
        self.config = SamplerConfig(chains=2,warmup=8,draws=10,seed=77,checkpoint_every=3)

    def fit(self,prior=None,covariance=None,config=None,**kwargs):
        return sample_regression(self.Y,self.Phi,self.Z,self.coordinates,self.groups,
                                 prior or PriorSpec.ridge(),covariance or CovarianceSpec(),
                                 config or self.config,**kwargs)

    def test_all_prior_families_return_finite_joint_draws(self):
        specs = [PriorSpec.ridge(),PriorSpec.grouped(),PriorSpec.group_rhs(),
                 PriorSpec.column_rhs(),PriorSpec.entry_rhs(),PriorSpec.gigg(),
                 PriorSpec.reference_r2(),PriorSpec.reference_r2(allocation="logistic_normal"),
                 PriorSpec.correction(base_columns=2),PriorSpec.function_penalty(base_columns=2)]
        config = replace(self.config,chains=1,warmup=3,draws=4)
        for prior in specs:
            with self.subTest(kind=prior.kind,unit=prior.unit,allocation=prior.allocation):
                posterior = self.fit(prior=prior,config=config)
                draw = posterior.draw_parameters(0,0,np.random.default_rng(99))
                self.assertEqual(draw.C.shape,(2,4))
                self.assertGreater(draw.sigma2,0)
                self.assertTrue(np.isfinite(posterior.log_density).all())
                self.assertEqual(posterior.chains.shape[:2],(1,4))
                self.assertEqual(posterior.chains.shape[2],len(posterior.parameter_names))

    def test_full_residual_ar_and_separate_scale_recovery(self):
        config = replace(self.config,chains=1,temporal="full_residual_ar")
        posterior = self.fit(config=config)
        draw = posterior.draw_parameters(0,2,np.random.default_rng(1))
        self.assertLess(abs(draw.temporal_parameters["rho"]),1)
        self.assertEqual(posterior.metadata["backend"],"temporal_whitened_spectral")
        pc = self.fit(covariance=CovarianceSpec(range_prior="pc",scale_mode="separate"),
                      config=replace(self.config,chains=1))
        sample = pc.draw_parameters(0,0,np.random.default_rng(2))
        self.assertEqual(sample.sigma2,np.exp(2*sample.covariance_parameters["log_sigma_e"]))
        self.assertEqual(pc.metadata["scale_integration"],"coefficients_only")

    def test_smoothness_categorical_update(self):
        cov = CovarianceSpec(infer_nu=True)
        fit = self.fit(covariance=cov)
        draws = [s["covariance_parameters"]["nu"] for chain in fit.states for s in chain]
        self.assertTrue(set(draws).issubset(set(cov.smoothness_choices)))
        self.assertIn("covariance.nu",fit.parameter_names)

    def test_checkpoint_resume_is_bitwise_and_checks_identity(self):
        full = self.fit()
        captured = {}
        class Interrupted(Exception): pass
        def callback(chain,state):
            captured[chain] = copy.deepcopy(state)
            if chain==0 and state["iteration"]==12:
                raise Interrupted()
        with self.assertRaises(Interrupted):
            self.fit(checkpoint_callback=callback)
        resumed = self.fit(resume_state=captured)
        assert_array_equal(full.chains,resumed.chains)
        assert_array_equal(full.log_density,resumed.log_density)
        assert_array_equal(full.acceptance,resumed.acceptance)
        self.assertEqual(captured[0]["proposal"]["count"],self.config.warmup)
        bad = copy.deepcopy(captured)
        bad[0]["signature"] = "wrong-data"
        with self.assertRaises(ValueError):
            self.fit(resume_state=bad)

    def test_conditional_checkpoint_and_extended_draws(self):
        config = replace(self.config,chains=1,algorithm="conditional")
        checkpoints = {}
        small = self.fit(prior=PriorSpec.entry_rhs(),config=replace(config,draws=4),
                         checkpoint_callback=lambda c,s:checkpoints.update({c:copy.deepcopy(s)}))
        extended = self.fit(prior=PriorSpec.entry_rhs(),config=config,resume_state=checkpoints)
        full = self.fit(prior=PriorSpec.entry_rhs(),config=config)
        assert_array_equal(extended.chains,full.chains)
        assert_array_equal(extended.log_density,full.log_density)
        assert_array_equal(small.chains,full.chains[:,:4])

    def test_safe_posterior_roundtrip_reproduces_recovery(self):
        posterior = self.fit()
        restored = RegressionPosterior.from_state(posterior.to_state(),self.Y,self.Phi,self.Z,
            self.coordinates,self.groups,posterior.prior,posterior.covariance,self.config)
        original = posterior.draw_parameters(1,3,np.random.default_rng(181))
        recovered = restored.draw_parameters(1,3,np.random.default_rng(181))
        assert_array_equal(original.C,recovered.C)
        self.assertEqual(original.sigma2,recovered.sigma2)

    def test_canonical_json_does_not_reorder_temporal_diagnostics(self):
        config = replace(self.config,chains=1,temporal="full_residual_ar")
        posterior = self.fit(config=config)
        payload = json.loads(json.dumps(posterior.to_state(),sort_keys=True,
                            default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item()))
        restored = RegressionPosterior.from_state(payload,self.Y,self.Phi,self.Z,self.coordinates,
                         self.groups,posterior.prior,posterior.covariance,config)
        self.assertEqual(restored.parameter_names,posterior.parameter_names)
        assert_array_equal(restored.chains,posterior.chains)

    def test_collapsed_rhs_tradeoff_checkpoint_continuation(self):
        config = replace(self.config,chains=1,draws=8,rhs_tradeoff_scale=.5)
        prior = PriorSpec.group_rhs()
        full = self.fit(prior=prior,config=config)
        saved = {}
        class Interrupted(Exception): pass
        def callback(chain,state):
            saved[chain] = copy.deepcopy(state)
            if state["iteration"]==12:
                raise Interrupted()
        with self.assertRaises(Interrupted):
            self.fit(prior=prior,config=config,checkpoint_callback=callback)
        restored = self.fit(prior=prior,config=config,resume_state=saved)
        assert_array_equal(restored.chains,full.chains)
        assert_array_equal(restored.log_density,full.log_density)
        self.assertEqual(restored.metadata["rhs_tradeoff_scale"],.5)

    def test_bounded_map_initialization_recovers_known_local_curvature(self):
        mean = np.array([1.3,-.8,.4])
        precision = np.array([[3.,.4,.1],[.4,1.5,-.2],[.1,-.2,2.]])
        def target(x):
            delta = x-mean
            return -.5*float(delta@precision@delta)
        config = SamplerConfig(init_maxiter=30,init_max_evaluations=200)
        calibration = _calibrate_initialization(target,np.zeros(3),config)
        # Initialization stops at a 1e-4 gradient norm, rather than pretending
        # that a local mode needs machine-precision optimization.
        assert_allclose(calibration["mode"],mean,atol=2e-5)
        assert_allclose(calibration["covariance"],np.linalg.inv(precision),atol=1e-9)
        self.assertLessEqual(calibration["optimizer_evaluations"],200)
        self.assertEqual(calibration["curvature_evaluations"],1+2*3**2)
        self.assertFalse(calibration["curvature_regularized"])
        limited = _calibrate_initialization(target,np.zeros(3),replace(config,init_max_evaluations=2))
        self.assertLessEqual(limited["optimizer_evaluations"],2)
        self.assertFalse(limited["optimizer_success"])

    def test_collapsed_chain_agrees_with_independent_hyperparameter_quadrature(self):
        Y = np.array([[.2,.7,-.3,.4]])
        Z = np.array([[1.],[-.4],[.2],[.7]])
        Phi = np.ones((1,1))
        prior = PriorSpec.ridge(mean_log_precision=0.,sd_log_precision=1.)
        eta = np.linspace(-7,7,12001)
        y,x = Y[0],Z[:,0]
        lam = np.exp(eta)
        Q = y@y-(x@y)**2/(lam+x@x)
        log_weights = -.5*np.log1p((x@x)/lam)-(2+len(y)/2)*np.log(.05+.5*Q)-.5*eta**2
        weights = np.exp(log_weights-special.logsumexp(log_weights))
        mean = float(weights@eta)
        variance = float(weights@np.square(eta-mean))
        config = SamplerConfig(chains=2,warmup=700,draws=3000,seed=172,proposal_scale=.4)
        fit = sample_regression(Y,Phi,Z,np.zeros((1,2)),["intercept"],prior,FixedIdentityCovariance(),config)
        sampled = fit.chains[:,:,0].reshape(-1)
        self.assertLess(abs(sampled.mean()-mean),.13)
        self.assertLess(abs(sampled.var()-variance),.17)


if __name__=="__main__":
    unittest.main()
