"""Book-grounded, deliberately LIMITED physics of stopping charged particles.

A. V. Belyaev, JINR P1-2023-22, equations (2), (7), (9), (14), table 1.
This is an analytical prior for SIMULATION/REVIEW, not a universal track
classifier. Valid only near a confirmed stop with explicit mean Bz and medium.
No fictitious expert pixel annotations and no real-film mass certifications.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize_scalar

W = 2.0 / 7.0
# Paper's table 1: fitted kappa/d are specific to its experimental model.
MEDIA = {
    "liquid_hydrogen": {"kappa_g_per_mev_cm2": 0.22, "density_g_per_cm3": 0.0585},
    "liquid_propane": {"kappa_g_per_mev_cm2": 0.45, "density_g_per_cm3": 0.427},
}


def _coefficients() -> np.ndarray:
    return np.asarray([1/(1-W), -1/(4*(1+W)), 3/(32*(3*W+1)),
                       -5/(128*(5*W+1)), 35/(2048*(7*W+1))], dtype=float)


def _validate(mean_hz_kgauss: float, medium: str, charge_sign: int):
    if medium not in MEDIA:
        raise ValueError("medium must be paper-calibrated liquid_hydrogen/liquid_propane")
    if not (-40 < float(mean_hz_kgauss) < -0.01):
        raise ValueError("paper's Hz<0 convention required: mean_hz_kgauss must be negative")
    if charge_sign not in (-1,1):
        raise ValueError("charge_sign must be +/-1")


def critical_length_cm(mass_mev: float, alpha_rad: float, medium: str) -> float:
    """a=(kappa*m/d)*cos(alpha), paper eq. (14), not a general stopping model."""
    if medium not in MEDIA or not (math.isfinite(mass_mev) and mass_mev>0):
        raise ValueError("finite positive mass and known medium required")
    if not (math.isfinite(alpha_rad) and abs(alpha_rad)<math.pi/2-0.05):
        raise ValueError("alpha must be a 3D-measured angle to the XY plane")
    props=MEDIA[medium]
    return props['kappa_g_per_mev_cm2']*mass_mev/props['density_g_per_cm3']*math.cos(alpha_rad)


def phi_from_s_cm(s_cm: np.ndarray, a_cm: float, *, medium: str, mean_hz_kgauss: float,
                  charge_sign: int) -> np.ndarray:
    """Full truncated five-term hadroida angular series, eq. (9), paper's range s/a<1.

    Only within a near-stop interval with averaged Bz. Projections onto a real
    camera require validated optics; this does NOT project onto BMP pixels.
    """
    _validate(mean_hz_kgauss,medium,charge_sign)
    s=np.asarray(s_cm,dtype=float)
    if s.ndim != 1 or s.size<2 or not np.isfinite(s).all() or np.any(s<0):
        raise ValueError('s must be a 1-D vector of finite nonnegative projected runs')
    if not math.isfinite(a_cm) or a_cm<=0 or np.any(s>=a_cm):
        raise ValueError("physical domain violation: require 0 <= s < a")
    props=MEDIA[medium]
    h0 = props['density_g_per_cm3']*math.sqrt(2)/(0.3*props['kappa_g_per_mev_cm2'])
    c=_coefficients()
    u=s/a_cm
    # c0*(1 + C1*u^(2w) + ...), where C_i=c_i/c0.
    power=np.stack([u**(2*W*i) for i in range(5)],axis=0)
    return charge_sign*(abs(mean_hz_kgauss)/h0)*(u**(1-W))*(c[:,None]*power).sum(axis=0)


@dataclass(frozen=True)
class SyntheticStoppingTrack:
    s_cm: np.ndarray
    xy_cm: np.ndarray
    phi_rad: np.ndarray
    a_cm: float
    source_model: str
    is_real_photo_label: bool=False


def simulate_stopping_track(*, mass_mev: float, alpha_rad: float, medium: str,
                            mean_hz_kgauss: float, charge_sign: int,
                            max_s_over_a: float = 0.65, points: int = 512) -> SyntheticStoppingTrack:
    if not (0.03 <= max_s_over_a < 1) or not 64<=points<=10000:
        raise ValueError('max_s_over_a must be <1 and points>=64')
    a=critical_length_cm(mass_mev,alpha_rad,medium)
    s=np.linspace(0, a*max_s_over_a, points)
    phi=phi_from_s_cm(s,a,medium=medium,mean_hz_kgauss=mean_hz_kgauss,charge_sign=charge_sign)
    x=cumulative_trapezoid(np.cos(phi),s,initial=0)
    y=cumulative_trapezoid(np.sin(phi),s,initial=0)
    return SyntheticStoppingTrack(s,np.column_stack((x,y)),phi,a,
                                 'BELYAEV_P1_2023_22_EQ9_SYNTHETIC_NOT_CAMERA_CALIBRATED')


def fit_hadroide_a(*, s_cm:np.ndarray, observed_phi_rad:np.ndarray, medium:str,
                   mean_hz_kgauss:float, charge_sign:int,
                   stopping_confirmed:bool, stereo_calibrated:bool,
                   optical_calibration_id:str, field_map_id:str,
                   alpha_rad:float, max_a_cm:float) -> dict:
    """Optional *review* fit to stereo 3D near-stop s,phi. Never infer stop from a line tip.

    Returns an approximate paper-model mass PROXY, NOT a particle ID. Model
    applicability and calibration IDs must be supplied explicitly, not guessed.
    """
    if not stopping_confirmed or not stereo_calibrated or not optical_calibration_id or not field_map_id:
        return {'status':'ABSTAIN_MISSING_EXPERT_STOP_OR_3D_CALIBRATION','physical_mass_certified':False}
    _validate(mean_hz_kgauss,medium,charge_sign)
    s=np.asarray(s_cm,float); p=np.asarray(observed_phi_rad,float)
    if s.shape != p.shape or s.ndim!=1 or len(s)<7 or not np.isfinite(s).all() or not np.isfinite(p).all() or np.any(s<0) or np.any(np.diff(s)<=0):
        raise ValueError('>=7 ordered, strictly increasing near-stop projected runs and angles required')
    critical_length_cm(1,alpha_rad,medium) # validate angle
    min_a=float(s[-1]/0.95)
    if not math.isfinite(max_a_cm) or max_a_cm<=min_a:
        return {'status':'ABSTAIN_NO_PHYSICALLY_ALLOWED_FIT_RANGE','physical_mass_certified':False}
    def residual(a):
        predicted=phi_from_s_cm(s,a,medium=medium,mean_hz_kgauss=mean_hz_kgauss,charge_sign=charge_sign)
        return float(np.mean((predicted-p)**2))
    opt=minimize_scalar(residual,bounds=(min_a,float(max_a_cm)),method='bounded',options={'xatol':1e-8})
    fitted_a=float(opt.x)
    props=MEDIA[medium]
    mass_proxy=fitted_a*props['density_g_per_cm3']/(props['kappa_g_per_mev_cm2']*math.cos(alpha_rad))
    return {'status':'PHYSICS_PRIOR_FIT_FOR_REVIEW', 'physical_mass_certified':False,
            'model':'BELYAEV_HADROIDA_NEAR_STOP_EQ9', 'a_cm':round(fitted_a,8),
            'approx_mass_proxy_mev':round(mass_proxy,6),
            'angular_rmse_rad':round(math.sqrt(opt.fun),8),
            'constraints':'s/a<1; near actual stopping point; averaged Bz; approximate energy-loss model',
            'optical_calibration_id':optical_calibration_id,'field_map_id':field_map_id,
            'fit_converged':bool(opt.success)}
