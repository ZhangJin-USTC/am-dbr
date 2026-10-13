"""v1.0.0 release-candidate rerun; frozen settings, no model selection.

Each completed cell has a source/config fingerprint. --resume accepts only an
identical run. Output arrays retain all five interval levels, PITs and scores.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(key,'1')
import argparse, concurrent.futures, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'framework'),str(ROOT/'scripts'),str(ROOT/'reproduction')]
from experiment import load_dataset, run_comparison, summary_table, multilevel_coverage_table
from fixed_adaptive_refresh import synthetic_dataset
from adaptive_memory_realdata_study import methods_factory, DATASETS, ALPHAS
from paper_config import PAPER_CONFIG
from revision_integrity import (canonical, cell_protocol, completed_payload_names,
    digest_json, method_payload_names, method_receipt_name, prepare_cell,
    read_json, require_equal, resume_method, verify_complete, write_json, write_receipt)
SCENARIOS=('no_shift','location','gradual','cycle','bursty','abrupt')


# This is an explicit, reviewable manifest.  Compatibility notebooks and
# legacy exploratory scripts remain in the public candidate package, but are
# intentionally outside the locked-results fingerprint unless listed here.
SOURCE_MANIFEST_FILES = (
    *(f"framework/{name}" for name in (
        "ACI_new_baselines.py", "cdf_tools.py", "core.py", "experiment.py",
        "experimental_methods.py", "make_figures.py", "make_notebooks.py",
        "methods.py", "metrics.py", "paper_config.py", "residual_baselines.py",
    )),
    "real_data/data_sources.py", "real_data/loaders.py",
    "reproduction/provenance.py", "reproduction/revision_integrity.py",
    "reproduction/reference_integrity.py",
    "scripts/run_revision.py", "scripts/fixed_adaptive_refresh.py",
    "scripts/adaptive_memory_realdata_study.py", "scripts/validate_upstream.py",
    "revision_tools/build_revision_artifacts.py",
    "revision_tools/run_state_sensitivity.py",
    "revision_tools/run_report_window_sensitivity.py",
    "revision_tools/validate_report_windows.py",
)
PUBLIC_COMPATIBILITY_FILES = (
    "scripts/rerun_main_realdata_fixed_adaptive.py",
    "framework/experimental_methods.py",
)


def fingerprint_entries() -> dict[str, str]:
    """Return the explicit source manifest used by a locked run.

    The previous implementation hashed only top-level files in three
    directories.  That omitted result validators and sensitivity generators,
    so a run could be resumed after changing those files.  This manifest is
    deliberately enumerated, so the clean public package has the same
    fingerprint even when unused legacy scripts are omitted.  A listed file
    must exist; changing it requires a deliberate rerun.
    """
    entries = {}
    for relative in SOURCE_MANIFEST_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required manifest source is missing: {path}")
        entries[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not entries:
        raise RuntimeError("Source manifest is empty")
    return entries


def fingerprint():
    entries = fingerprint_entries()
    return hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()


def source_record():
    entries = fingerprint_entries()
    return {'source_sha256': hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest(),
            'source_files': sorted(entries), 'source_file_hashes': entries}


def expected_cell(kind, name, seed, *, dataset_spec=None, source_sha=None):
    if kind not in ('synthetic', 'real'):
        raise ValueError(f'Unknown study kind: {kind}')
    spec = None
    if kind == 'real':
        spec = dict(dataset_spec or next(s for s in DATASETS if s['label'] == name))
    config = {'paper_config': PAPER_CONFIG.as_dict(), 'alphas': ALPHAS,
              'methods': list(methods_factory()), 'dataset_spec': spec,
              'base_model': 'ridge' if kind == 'real' else 'frozen synthetic forecast',
              'scoring': 'exact integrated squared PIT-CDF error'}
    identity = {'kind': kind, 'name': name, 'seed': int(seed)}
    if spec is not None:
        identity['report_start_fraction'] = spec['eval_start_frac']
    return cell_protocol(source_sha or fingerprint(), identity, config)


def main_protocol(seeds, scenarios, datasets):
    config = {'paper_config': PAPER_CONFIG.as_dict(), 'alphas': ALPHAS,
              'seeds': list(seeds), 'scenarios': list(scenarios),
              'datasets': [dict(next(s for s in DATASETS if s['label']==name)) for name in datasets],
              'methods': list(methods_factory()), 'base_model': 'ridge'}
    return canonical({'version':'1.0.0-rc.1',
        'status':'corrective rerun of previously inspected data; not a new blinded test',
        **source_record(), 'study_config': config, 'config_sha256':digest_json(config),
        'config':PAPER_CONFIG.as_dict(),'alphas':ALPHAS,'seeds':seeds,
        'scenarios':scenarios,'datasets':datasets,'scoring':'exact integrated squared PIT-CDF error',
        'bins_comparison':'140 bins for all binned methods; AM stores seven experts, OCC two arrays',
        'parent_version':'0.9.5','selection':'no retuning; historical frozen AM library',
        'time_preprocessing':'sort/mean duplicates/exact-time lags/drop missing; local source clock'})


def run_cell(task):
    kind,name,seed,out,cache=task
    started=time.perf_counter();dest=Path(out)/kind/(f'{name}_{seed}' if kind=='synthetic' else name)
    expected = expected_cell(kind, name, seed)
    names = list(methods_factory())
    prepare_cell(dest, expected)
    if (dest/'COMPLETE.json').exists():
        verify_complete(dest, expected, completed_payload_names(names))
        return str(dest)
    if kind=='synthetic':
        dataset=synthetic_dataset(name,seed)
    else:
        spec=next(s for s in DATASETS if s['label']==name)
        dataset=load_dataset(spec['name'],data_dir=cache,max_eval=spec['max_eval'],model_type='ridge',
                             train_frac=spec['train_frac'],warm_frac=spec['warm_frac'],eval_start_frac=spec['eval_start_frac'])
    y=np.array([s.y for s in dataset['online']])
    summaries=[]
    for method_name in names:
        # Per-method checkpoints make long real-data initialization recoverable.
        csv=dest/f'{method_name}.csv'; npz=dest/f'{method_name}.npz'
        if resume_method(dest, method_name, expected):
            summaries.append(pd.read_csv(csv));continue
        print(f'START {kind}/{name}/{seed} {method_name}',flush=True)
        result=run_comparison(dataset,alphas=ALPHAS,methods_factory=lambda:{method_name:methods_factory()[method_name]})
        summary=summary_table(result,y,alpha_main=.1,rolling_window=500,round_digits=None).reset_index()
        multi=multilevel_coverage_table(result,y,alphas=ALPHAS,round_digits=None).reset_index()
        summary=summary.merge(multi,on='method',validate='one_to_one')
        summary.insert(0,'seed',seed);summary.insert(0,'scenario' if kind=='synthetic' else 'data_set',name)
        r=result[method_name]
        arrays={'y':y,'t':np.array([s.t for s in dataset['online']]),'u':np.array([s.u for s in dataset['online']]),'alphas':np.array(ALPHAS),
                'lower':np.stack([r.intervals_by_alpha[a][0] for a in ALPHAS]),
                'upper':np.stack([r.intervals_by_alpha[a][1] for a in ALPHAS])}
        if r.calibrated_pit is not None:arrays.update(calibrated_pit=r.calibrated_pit,pit_crps=r.pit_crps)
        with npz.with_suffix('.npz.part').open('wb') as f:np.savez_compressed(f,**arrays)
        npz.with_suffix('.npz.part').replace(npz)
        if r.diagnostics:(dest/f'{method_name}_diagnostics.json').write_text(json.dumps(r.diagnostics,indent=2),encoding='utf-8')
        summary.to_csv(csv.with_suffix('.csv.part'),index=False);csv.with_suffix('.csv.part').replace(csv)
        write_receipt(dest, method_receipt_name(method_name), expected,
                      method_payload_names(method_name), method=method_name)
        summaries.append(summary)
    pd.concat(summaries,ignore_index=True).to_csv(dest/'summary.csv',index=False)
    write_json(dest/'metadata.json', dataset['meta'])
    write_receipt(dest, 'COMPLETE.json', expected, completed_payload_names(names),
                  elapsed_s=time.perf_counter()-started, methods=names)
    return str(dest)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--workers',type=int,default=3);p.add_argument('--resume',action='store_true')
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(6100,6108)))
    p.add_argument('--datasets',nargs='*',default=[s['label'] for s in DATASETS]);p.add_argument('--scenarios',nargs='*',default=list(SCENARIOS))
    args=p.parse_args()
    if args.workers<1: p.error('--workers must be positive')
    if any(len(values)!=len(set(values)) for values in (args.datasets,args.scenarios,args.seeds)):
        raise ValueError('Duplicate dataset/scenario/seed would schedule the same output cell concurrently')
    if set(args.datasets)-{s['label'] for s in DATASETS} or set(args.scenarios)-set(SCENARIOS):raise ValueError('Unknown study cell')
    protocol=main_protocol(args.seeds,args.scenarios,args.datasets);file=args.out/'protocol.json'
    if args.out.exists() and any(args.out.iterdir()):
        if not args.resume or not file.exists() or json.loads(file.read_text())!=protocol:
            raise FileExistsError('Nonempty output requires --resume and an identical source/config fingerprint.')
    args.out.mkdir(parents=True,exist_ok=True);file.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    tasks=[('real',s,-1,str(args.out),str(args.data_dir)) for s in args.datasets]
    tasks += [('synthetic',s,seed,str(args.out),str(args.data_dir)) for s in args.scenarios for seed in args.seeds]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_cell,t) for t in tasks]
        for future in concurrent.futures.as_completed(futures): print('COMPLETE',future.result(),flush=True)
    for kind in ['synthetic','real']:
        files=sorted((args.out/kind).glob('*/summary.csv'))
        if files:pd.concat([pd.read_csv(f) for f in files],ignore_index=True).to_csv(args.out/f'{kind}_summary.csv',index=False)
    print('Revision run complete.',flush=True)


if __name__=='__main__':main()
