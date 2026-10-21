"""Predeclared count-state sensitivity, separate from parameter selection."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'1')
from pathlib import Path
import sys,argparse,concurrent.futures,hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'framework'),str(ROOT/'scripts')]
from methods import OCC,DirectBinnedRecalibration
from fixed_adaptive_refresh import synthetic_dataset
from experiment import run_comparison,summary_table,multilevel_coverage_table
from adaptive_memory_realdata_study import ALPHAS
from paper_config import PAPER_CONFIG
from run_revision import SCENARIOS, fingerprint, source_record
from revision_integrity import (canonical, cell_protocol, completed_payload_names,
    method_payload_names, method_receipt_name, prepare_cell, verify_complete,
    write_json, write_receipt)

STATE_CONFIG = {
    'status': 'declared count-state sensitivity; no retuning',
    'seeds': list(range(6100, 6108)),
    'scenarios': list(SCENARIOS),
    'alphas': list(ALPHAS),
    'synthetic_config': PAPER_CONFIG.as_dict(),
    'comparison_1': 'DBR140 versus OCC70: 140 count/sum accumulators',
    'comparison_2': 'DBR987 versus AM7x140+7weights: 987 accumulators; cached arrays not equalised',
    'methods': {'OCC-Fixed-B70': {'lam': .985, 'bins': 70},
                'DBR-Fixed-B987': {'lam': .985, 'bins': 987}},
}


def config_fingerprint() -> str:
    payload = json.dumps(STATE_CONFIG, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def expected_state_cell(scenario, seed, source_sha=None):
    return cell_protocol(source_sha or fingerprint(),
                         {'kind':'state_sensitivity','scenario':scenario,'seed':int(seed)},
                         STATE_CONFIG)


def state_protocol():
    return canonical(dict(STATE_CONFIG, schema_version=1, **source_record(),
                          config_sha256=config_fingerprint()))

def cell(task):
    scenario,seed,out,source_sha,config_sha=task
    if source_sha!=fingerprint() or config_sha!=config_fingerprint():
        raise ValueError('State source/config changed after scheduling')
    dest=Path(out)/f'{scenario}_{seed}'
    expected=expected_state_cell(scenario,seed,source_sha)
    methods=list(STATE_CONFIG['methods'])
    prepare_cell(dest,expected)
    if (dest/'COMPLETE.json').exists():
        verify_complete(dest,expected,completed_payload_names(methods))
        return f'{scenario}/{seed}'
    # This runner commits a whole cell. Partial final files cannot be reused.
    if any(dest.glob('*.csv')) or any(dest.glob('*.npz')):
        raise ValueError(f'Interrupted state cell requires a fresh output directory: {dest}')
    ds=synthetic_dataset(scenario,seed);y=np.array([s.y for s in ds['online']])
    result=run_comparison(ds,ALPHAS,lambda:{'OCC-Fixed-B70':OCC(**STATE_CONFIG['methods']['OCC-Fixed-B70']),'DBR-Fixed-B987':DirectBinnedRecalibration(**STATE_CONFIG['methods']['DBR-Fixed-B987'])})
    df=summary_table(result,y,alpha_main=.1,rolling_window=500,round_digits=None).reset_index().merge(multilevel_coverage_table(result,y,ALPHAS,round_digits=None).reset_index(),on='method')
    df.insert(0,'seed',seed);df.insert(0,'scenario',scenario)
    df.to_csv(dest/'summary.csv',index=False)
    for name,r in result.items():
        np.savez_compressed(dest/f'{name}.npz',y=y,alphas=ALPHAS,lower=np.stack([r.intervals_by_alpha[a][0] for a in ALPHAS]),upper=np.stack([r.intervals_by_alpha[a][1] for a in ALPHAS]),pit_crps=r.pit_crps,calibrated_pit=r.calibrated_pit)
        df[df.method==name].to_csv(dest/f'{name}.csv',index=False)
        write_receipt(dest,method_receipt_name(name),expected,method_payload_names(name),method=name)
    write_json(dest/'metadata.json',ds['meta'])
    write_receipt(dest,'COMPLETE.json',expected,completed_payload_names(methods),methods=methods)
    return f'{scenario}/{seed}'

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--workers',type=int,default=3);args=p.parse_args()
    if args.workers<1:p.error('--workers must be positive')
    if args.out.exists() and any(args.out.iterdir()):raise FileExistsError('Output must be empty')
    args.out.mkdir(parents=True,exist_ok=True)
    source_sha = fingerprint()
    config_sha = config_fingerprint()
    protocol = state_protocol()
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    tasks=[(s,i,str(args.out),source_sha,config_sha) for s in SCENARIOS for i in range(6100,6108)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for msg in ex.map(cell,tasks):print(msg,flush=True)
    pd.concat([pd.read_csv(f) for f in sorted(args.out.glob('*/summary.csv'))],ignore_index=True).to_csv(args.out/'summary.csv',index=False)

if __name__=='__main__':main()
