"""Bounded optimizer continuation with unchanged convergence tolerances."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def projected_gradient(parameters, gradient, bounds, tolerance=1e-8):
    """L-infinity KKT residual for a box-constrained smooth objective."""
    x=np.asarray(parameters,float);g=np.array(gradient,dtype=float,copy=True)
    limits=np.asarray(bounds,float)
    if x.shape!=g.shape or limits.shape!=(len(x),2) or not np.isfinite(g).all():
        return None
    g[(x<=limits[:,0]+tolerance)&(g>0)]=0.
    g[(x>=limits[:,1]-tolerance)&(g<0)]=0.
    return float(np.max(np.abs(g),initial=0.))


def bounded_lbfgsb(objective, initial, bounds, *, max_iter, ftol, gtol=1e-5,
                   continuation_max_iter=0):
    """At most two phases; continue only an improving iteration-limited fit.

    This is a declared restart of L-BFGS-B from its last point, not an exact
    continuation of the optimizer's private quasi-Newton memory. It never
    lowers tolerances or treats an exhausted budget as convergence.
    """
    for key,value,minimum in (("max_iter",max_iter,1),
                              ("continuation_max_iter",continuation_max_iter,0)):
        if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value<minimum:
            raise ValueError(f"{key} must be an integer >= {minimum}")
    for name,value in (("ftol",ftol),("gtol",gtol)):
        if not np.isfinite(value) or value<=0:
            raise ValueError(f"{name} must be positive and finite")
    phases=[];x=np.asarray(initial,float);initial_value=None;best=None;rejected=False
    # Only evaluate the additional initial objective when continuation is
    # enabled. The legacy single-phase numerical call remains unchanged.
    if continuation_max_iter:
        initial_value=float(objective(x))
    for index,cap in enumerate((max_iter,continuation_max_iter)):
        if cap==0:
            break
        result=minimize(objective,x,method="L-BFGS-B",bounds=bounds,
                        options={"maxiter":int(cap),"ftol":float(ftol),"gtol":float(gtol),"maxls":30})
        value=float(result.fun)
        finite=bool(np.isfinite(value) and value<1e99 and np.isfinite(result.x).all())
        gradient=projected_gradient(result.x,result.jac,bounds)
        phases.append({"phase":index,"max_iter":int(cap),"iterations":int(result.nit),
                       "function_evaluations":int(result.nfev),"success":bool(result.success),
                       "status":int(result.status),"message":str(result.message),
                       "objective":value if finite else None,"projected_gradient_inf":gradient,
                       "ftol":float(ftol),"gtol":float(gtol)})
        if best is None:
            best=result
        elif finite and value<=float(best.fun)+1e-10*(1+abs(float(best.fun))):
            best=result
        else:
            rejected=True
        if index or result.success or not continuation_max_iter:
            break
        improving=(finite and initial_value is not None and np.isfinite(initial_value)
                   and value<initial_value-1e-12*(1+abs(initial_value)))
        iteration_limited=(int(result.status)==1 and int(result.nit)>=max_iter)
        if not (iteration_limited and improving):
            break
        x=np.array(result.x,copy=True)
    metadata={"phases":phases,"continuations":len(phases)-1,
              "iterations":sum(item["iterations"] for item in phases),
              "function_evaluations":sum(item["function_evaluations"] for item in phases),
              "initial_objective_evaluations":int(bool(continuation_max_iter)),
              "rejected_worsening_continuation":rejected,
              "qualified":bool(best.success and not rejected and np.isfinite(best.fun)
                                and float(best.fun)<1e99 and np.isfinite(best.x).all()),
              "continuation_policy":"one improving iteration-cap restart; unchanged tolerances",
              "projected_gradient_inf":projected_gradient(best.x,best.jac,bounds)}
    return best,metadata
