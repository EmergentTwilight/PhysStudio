"""Plot phase progression and energy balance from cached simulation states."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .thermal import LATENT, fraction, temperature


def summarize(path):
    cfg=json.loads((path/'config.json').read_text())
    audits=json.loads((path/'audit.json').read_text())
    with np.load(path/'initial.npz') as initial:
        original_ice=initial['groups']>=0
    rows=[]
    for audit in audits:
        with np.load(path/'states'/f"{audit['frame']:05d}.npz") as state:
            e=state['enthalpy'][original_ice]
            rows.append(dict(time=float(state['time']),mean_temperature=float(temperature(e).mean()),
                mean_liquid_fraction=float(fraction(e).mean()),
                released_fraction=float(np.mean(e>=LATENT)),
                energy_balance_J=audit['energy_balance_J'],external_heat_J=audit['external_heat_J']))
    return dict(scene=cfg['scene'],label=cfg['label'],thermal_scale=cfg['thermal_scale'],
        particles=cfg['particles'],initial_ice_particles=int(original_ice.sum()),
        initial_ice_mass_kg=int(original_ice.sum())*cfg['mass'],rows=rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--simulations',nargs='+',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    results=[summarize(path) for path in args.simulations]
    (args.output/'thermal_history.json').write_text(json.dumps(results,indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(2,len(results),figsize=(5*len(results),6.4),squeeze=False)
    for col,result in enumerate(results):
        rows=result['rows'];t=np.array([r['time'] for r in rows])
        a=axes[0,col]
        a.plot(t,[100*r['mean_liquid_fraction'] for r in rows],color='#207e9a',lw=2,label='Latent-heat fraction')
        a.plot(t,[100*r['released_fraction'] for r in rows],color='#d07a3a',lw=2,label='Released liquid particles')
        a.set(xlabel='Mechanical time (s)',ylabel='Initial ice converted (%)',ylim=(-2,102),
            title=f"{result['label']}  ·  heat clock ×{result['thermal_scale']:g}")
        a.legend(frameon=False,fontsize=9,loc='upper left');a.grid(alpha=.16)
        a=axes[1,col]
        a.plot(t,[r['energy_balance_J'] for r in rows],color='#5d668c',lw=2)
        a.axhline(0,color='#555555',lw=.7)
        a.set(xlabel='Mechanical time (s)',ylabel='Energy-balance residual (J)');a.grid(alpha=.16)
    fig.suptitle('Ice melting: phase progression and heat accounting',fontsize=17,y=1.01)
    fig.tight_layout();fig.savefig(args.output/'thermal_progression.png',dpi=150,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps([dict(scene=r['scene'],final=r['rows'][-1]) for r in results],indent=2))


if __name__=='__main__':
    main()
