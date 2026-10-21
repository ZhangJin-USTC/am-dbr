"""Preserve interrupted, uncommitted method files before a frozen runner resumes.

Committed checkpoints and whole cells must pass integrity verification. Only
uncommitted payloads in an otherwise matching cell may be moved and recomputed.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/p) for p in ('framework','scripts','reproduction','revision_tools')]
from revision_integrity import (completed_payload_names,method_payload_names,
    method_receipt_name,read_json,require_equal,verify_complete,verify_receipt)


def recover_cell(cell,expected,methods,quarantine):
    cell=Path(cell)
    if not cell.exists() or not any(cell.iterdir()):return []
    require_equal(read_json(cell/'CELL_PROTOCOL.json'),expected,'Partial cell protocol')
    if (cell/'COMPLETE.json').exists():
        verify_complete(cell,expected,completed_payload_names(methods))
        return []
    retained=[]
    for method in methods:
        receipt=method_receipt_name(method)
        names=method_payload_names(method)
        if (cell/receipt).exists():
            verify_receipt(cell,receipt,expected,names)
            continue
        candidates=[cell/name for name in names]
        candidates += [cell/(name+'.part') for name in names+[receipt]]
        for path in candidates:
            if path.exists():
                if not path.is_file():raise ValueError(f'Expected a payload file: {path}')
                destination=Path(quarantine)/method/path.name
                destination.parent.mkdir(parents=True,exist_ok=True)
                shutil.move(str(path),str(destination));retained.append(str(destination))
    return retained


def prepare(out,study):
    from run_revision import SCENARIOS,expected_cell,main_protocol
    from adaptive_memory_realdata_study import DATASETS,methods_factory
    from run_report_window_sensitivity import window_protocol
    out=Path(out)
    if not out.exists() or not any(out.iterdir()):return []
    protocol=read_json(out/'protocol.json')
    methods=list(methods_factory())
    cells=[]
    if study=='main':
        require_equal(protocol,main_protocol(list(range(6100,6108)),list(SCENARIOS),[s['label'] for s in DATASETS]),'Main protocol')
        cells += [(Path('synthetic')/f'{s}_{i}',expected_cell('synthetic',s,i)) for s in SCENARIOS for i in range(6100,6108)]
        cells += [(Path('real')/s['label'],expected_cell('real',s['label'],-1)) for s in DATASETS]
    else:
        require_equal(protocol,window_protocol(),'Window protocol')
        for fraction in protocol['report_start_fractions']:
            for spec in DATASETS:
                cells.append((Path(f'report_{round(fraction*100):03d}')/'real'/spec['label'],
                    expected_cell('real',spec['label'],-1,dataset_spec=dict(spec,eval_start_frac=fraction))))
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    retained=[]
    for relative,expected in cells:
        retained += recover_cell(out/relative,expected,methods,out/'_interrupted'/stamp/relative)
    for item in retained:print(f'Preserved uncommitted method payload: {item}',flush=True)
    return retained


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--study',choices=('main','windows'),required=True)
    args=parser.parse_args();prepare(args.out,args.study)
