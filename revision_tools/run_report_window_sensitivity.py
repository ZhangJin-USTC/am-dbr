"""Frozen-method report-window sensitivity, separate from the main study.

This reuses the main runner's output/checkpoint format. Dataset specifications
are changed only inside the worker process for the selected reporting fraction;
training, warm-up, methods, levels and caps are unchanged. Windows may overlap.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'framework'),str(ROOT/'scripts')]
from run_revision import run_cell, fingerprint, source_record
from revision_integrity import canonical, digest_json
from adaptive_memory_realdata_study import DATASETS, ALPHAS, methods_factory
from paper_config import PAPER_CONFIG

FRACTIONS=(.75,.85,.95)


def window_protocol():
    config={'report_start_fractions':FRACTIONS,'datasets':DATASETS,
            'paper_config':PAPER_CONFIG.as_dict(),'alphas':ALPHAS,
            'methods':list(methods_factory()),'base_model':'ridge'}
    return canonical({
        'version':'1.0.0-rc.1',
        'status':'additional report-window sensitivity after inspecting main corrective results; not a new blinded test',
        **source_record(),
        'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'study_config':config,'config_sha256':digest_json(config),
        'report_start_fractions':FRACTIONS,'datasets':DATASETS,
        'config':PAPER_CONFIG.as_dict(),'alphas':ALPHAS,
        'methods':list(methods_factory()),
        'unchanged':'training/warmup fractions, report caps, methods, parameters, source snapshots',
        'interpretation':'within-dataset sensitivity; windows may overlap; no IID confidence intervals or retuning',
    })

def worker(task):
    label,fraction,out,cache=task
    spec=next(s for s in DATASETS if s['label']==label)
    original=spec['eval_start_frac']
    # Process-local override. No parent configuration or shared file is changed.
    spec['eval_start_frac']=fraction
    try:
        dest=run_cell(('real',label,-1,str(Path(out)/f'report_{round(fraction*100):03d}'),cache))
    finally:
        spec['eval_start_frac']=original
    return label,fraction,dest

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be positive')
    protocol=window_protocol()
    path=args.out/'protocol.json'
    if args.out.exists() and any(args.out.iterdir()):
        if not args.resume or not path.exists() or json.loads(path.read_text(encoding='utf-8'))!=protocol:
            raise FileExistsError('Nonempty sensitivity output requires --resume and an identical protocol.')
    args.out.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    tasks=[(s['label'],f,str(args.out),str(args.data_dir)) for f in FRACTIONS for s in DATASETS]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(worker,task) for task in tasks]
        for job in concurrent.futures.as_completed(jobs):
            print('COMPLETE',job.result(),flush=True)
    frames=[]
    for label,fraction,_,_ in tasks:
        folder=args.out/f'report_{round(fraction*100):03d}'/'real'/label
        if not (folder/'COMPLETE.json').is_file():
            raise FileNotFoundError(folder/'COMPLETE.json')
        frame=pd.read_csv(folder/'summary.csv')
        if frame.method.duplicated().any() or set(frame.method)!=set(protocol['methods']):
            raise AssertionError(f'Window method schema mismatch: {folder}')
        frame.insert(0,'report_start_fraction',fraction)
        frames.append(frame)
    combined=pd.concat(frames,ignore_index=True)
    if len(combined)!=len(protocol['methods'])*len(tasks):
        raise AssertionError('Missing or duplicated sensitivity method rows')
    combined.to_csv(args.out/'summary.csv',index=False)
    from validate_report_windows import validate_windows
    validate_windows(args.out)
    print('Report-window sensitivity complete:',len(tasks),'cells,',len(combined),'method runs.',flush=True)

if __name__=='__main__':
    main()
