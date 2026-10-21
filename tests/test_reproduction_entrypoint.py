"""Public orchestration must reject corrupt metrics and preserve partial work."""
from pathlib import Path
import json
import sys
import subprocess
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/p) for p in ('framework','scripts','reproduction','revision_tools')]
from compare_reference import compare
import resume_state
from resume_checkpoints import recover_cell
from revision_integrity import write_receipt,method_payload_names


def test_reference_comparison_ignores_runtime_but_rejects_metric_changes():
    expected=pd.DataFrame({'method':['cdf','interval'],'score':[.15,np.nan],'width':[2.,np.inf]})
    actual=expected.assign(runtime_s=[999.,100.])
    assert compare(actual,expected,['method'])['rows']==2
    actual.loc[0,'score']+=.01
    with pytest.raises(AssertionError):
        compare(actual,expected,['method'])


def test_reference_comparison_rejects_duplicate_or_missing_identity():
    frame=pd.DataFrame({'method':['a','b'],'score':[.1,.2]})
    with pytest.raises(ValueError):
        compare(pd.concat([frame,frame.iloc[:1]]),frame,['method'])
    with pytest.raises(ValueError):
        compare(frame.iloc[:1],frame,['method'])


def test_state_resume_preserves_uncommitted_payload(tmp_path,monkeypatch):
    monkeypatch.setattr(resume_state,'SCENARIOS',('tiny',))
    monkeypatch.setattr(resume_state,'fingerprint',lambda:'source')
    monkeypatch.setattr(resume_state,'config_fingerprint',lambda:'config')
    protocol={'example':'fixed'}
    monkeypatch.setattr(resume_state,'state_protocol',lambda:protocol)
    out=tmp_path/'state';dest=out/'tiny_6100';dest.mkdir(parents=True)
    (out/'protocol.json').write_text(json.dumps(protocol))
    (dest/'uncommitted.npz').write_bytes(b'partial result')
    tasks=resume_state.prepare(out)
    assert len(tasks)==8
    retained=list((out/'_interrupted').glob('*/tiny_6100/uncommitted.npz'))
    assert len(retained)==1 and retained[0].read_bytes()==b'partial result'
    assert not dest.exists()


def test_state_resume_rejects_changed_protocol(tmp_path,monkeypatch):
    (tmp_path/'protocol.json').write_text('{"source":"wrong"}')
    monkeypatch.setattr(resume_state,'state_protocol',lambda:{'source':'expected'})
    with pytest.raises(ValueError):
        resume_state.prepare(tmp_path)


def test_method_resume_preserves_only_uncommitted_files(tmp_path):
    cell=tmp_path/'cell';cell.mkdir();expected={'source':'fixed'}
    (cell/'CELL_PROTOCOL.json').write_text(json.dumps(expected))
    (cell/'Base.npz').write_bytes(b'committed array')
    (cell/'Base.csv').write_text('committed summary')
    write_receipt(cell,'Base_CHECKPOINT.json',expected,method_payload_names('Base'))
    (cell/'Incomplete.npz').write_bytes(b'partial array')
    saved=recover_cell(cell,expected,['Base','Incomplete'],tmp_path/'quarantine')
    assert len(saved)==1 and Path(saved[0]).read_bytes()==b'partial array'
    assert (cell/'Base.npz').read_bytes()==b'committed array'
    assert not (cell/'Incomplete.npz').exists()
    (cell/'Base.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='hash mismatch'):
        recover_cell(cell,expected,['Base'],tmp_path/'quarantine2')


def test_failed_reference_rerun_replaces_stale_pass_receipt(tmp_path):
    report=tmp_path/'reference_comparison.json'
    report.write_text('{"status":"passed"}')
    result=subprocess.run([sys.executable,str(ROOT/'reproduction/compare_reference.py'),
        '--out',str(tmp_path)],capture_output=True,text=True)
    assert result.returncode!=0
    assert json.loads(report.read_text())['status']=='failed'
