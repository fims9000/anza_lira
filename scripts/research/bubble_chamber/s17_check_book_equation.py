#!/usr/bin/env python3
"""Self-consistency of the Belyaev synthetic physics prior, NOT independent physics validation."""
from pathlib import Path
import json
import numpy as np
import sys
ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from bubble_chamber.s17_belyaev_hadroide import simulate_stopping_track,fit_hadroide_a


def check(*,seed=20261025,n=40):
    rng=np.random.default_rng(seed);rows=[]
    for i in range(n):
        medium='liquid_hydrogen' if i%2==0 else 'liquid_propane'
        mass=float(rng.uniform(140,950));alpha=float(rng.uniform(-.4,.4))
        q=1 if i%3 else -1
        sim=simulate_stopping_track(mass_mev=mass,alpha_rad=alpha,medium=medium,
             mean_hz_kgauss=-18,charge_sign=q,max_s_over_a=.55,points=600)
        picks=np.linspace(0,599,24).astype(int)
        turn=sim.phi_rad[picks]+rng.normal(0,.03,len(picks))
        fit=fit_hadroide_a(s_cm=sim.s_cm[picks],observed_phi_rad=turn,
           medium=medium,mean_hz_kgauss=-18,charge_sign=q,
           stopping_confirmed=True,stereo_calibrated=True,
           optical_calibration_id='ARTIFICIAL_EXACT_KNOWN_STEREO',
           field_map_id='ARTIFICIAL_CONSTANT_BZ',alpha_rad=alpha,max_a_cm=sim.a_cm*2.5)
        if fit['status']!='PHYSICS_PRIOR_FIT_FOR_REVIEW':raise RuntimeError(f'bad fit #{i}')
        rows.append({'medium':medium,'mass_true_mev':mass,'mass_proxy_mev':fit['approx_mass_proxy_mev'],
                     'rel_error':abs(fit['approx_mass_proxy_mev']/mass-1),
                     'angular_rmse':fit['angular_rmse_rad']})
    e=np.asarray([r['rel_error'] for r in rows]);rms=np.asarray([r['angular_rmse'] for r in rows])
    return {'schema':'s17_book_equation_self_consistency_v1','is_independent_experimental_validation':False,
            'same_equation_used_for_generation_and_fit':True,
            'assumed_exact_B_and_optics':True,'synthetic_gaussian_phi_noise_rad':.03,
            'n_simulated_tracks':len(rows),'mediums':sorted(set(x['medium'] for x in rows)),
            'median_relative_mass_proxy_error':round(float(np.median(e)),6),
            'p90_relative_mass_proxy_error':round(float(np.quantile(e,.9)),6),
            'mean_angular_fit_rmse_rad':round(float(np.mean(rms)),6),
            'no_real_photo_labels':True,'not_particle_identification':True}


if __name__=='__main__':
    out=Path('/mnt/data/anza_s15/research_outputs/s17_book_prior/s17_equation_selfcheck.json')
    result=check();out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
