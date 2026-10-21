"""Validate every declared reporting window, cell receipt and summary value."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'framework'),str(ROOT/'scripts'),str(ROOT/'reproduction')]
from run_revision import expected_cell
from run_report_window_sensitivity import window_protocol
from build_revision_artifacts import validate_cell
from revision_integrity import (completed_payload_names, file_digest, read_json,
    require_equal, verify_complete, write_json)


def validate_windows(out):
    out=Path(out)
    protocol=read_json(out/'protocol.json')
    expected=window_protocol()
    require_equal(protocol,expected,'Report-window protocol')
    methods=protocol['methods']
    fractions=protocol['report_start_fractions']
    specs=protocol['datasets']
    expected_dirs={f'report_{round(f*100):03d}' for f in fractions}
    if {p.name for p in out.glob('report_*') if p.is_dir()}!=expected_dirs:
        raise ValueError('Missing or unexpected report-fraction directory')
    rows=[];cells=[];predictions=0
    for fraction in fractions:
        real=out/f'report_{round(fraction*100):03d}'/'real'
        if {p.name for p in real.iterdir() if p.is_dir()}!={s['label'] for s in specs}:
            raise ValueError(f'Missing or unexpected window dataset: {real}')
        for baseline_spec in specs:
            spec=dict(baseline_spec,eval_start_frac=fraction)
            name=spec['label'];dest=real/name
            provenance=expected_cell('real',name,-1,dataset_spec=spec,
                                     source_sha=protocol['source_sha256'])
            verify_complete(dest,provenance,completed_payload_names(methods))
            frame,n=validate_cell(dest,methods)
            if set(frame.data_set)!={name} or set(frame.seed)!={-1}:
                raise ValueError(f'Window row identity mismatch: {dest}')
            meta=read_json(dest/'metadata.json')
            expected_start=max(meta['warm_end'],
                               int(np.floor(meta['n_total']*fraction)))
            expected_end=min(meta['n_total'],expected_start+spec['max_eval'])
            if (meta['report_start']!=expected_start or meta['eval_end']!=expected_end
                    or meta['n_online']!=n or n!=expected_end-expected_start
                    or meta['eval_start_frac']!=fraction):
                raise ValueError(f'Window metadata/report index mismatch: {dest}')
            frame.insert(0,'report_start_fraction',fraction);rows.append(frame)
            cells.append({'dataset':name,'fraction':fraction,'n_report':n,
                          'first_timestamp':meta['report_first_timestamp'],
                          'last_timestamp':meta['report_last_timestamp'],
                          'complete_sha256':file_digest(dest/'COMPLETE.json')})
            predictions+=n*len(methods)*len(protocol['alphas'])
    combined=pd.concat(rows,ignore_index=True)
    actual=pd.read_csv(out/'summary.csv')
    if len(actual)!=len(fractions)*len(specs)*len(methods):
        raise ValueError('Window summary count mismatch')
    keys=['report_start_fraction','data_set','method']
    if actual.duplicated(keys).any():
        raise ValueError('Duplicate window summary identity')
    pd.testing.assert_frame_equal(actual.sort_values(keys).reset_index(drop=True),
                                  combined.sort_values(keys).reset_index(drop=True),
                                  check_exact=False,rtol=1e-11,atol=1e-12)
    report={'source_sha256':protocol['source_sha256'],'config_sha256':protocol['config_sha256'],
            'cells':cells,'method_runs':len(combined),'interval_predictions':predictions,
            'payload_hashes_verified':True,'array_summary_checks':'passed',
            'summary_sha256':file_digest(out/'summary.csv')}
    write_json(out/'window_validation.json',report)
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--windows',type=Path,required=True)
    args=parser.parse_args()
    report=validate_windows(args.windows)
    print(f"Validated {len(report['cells'])} windows, {report['method_runs']} method runs.")


if __name__=='__main__':main()
