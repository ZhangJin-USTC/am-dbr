"""Reject stale/mixed evidence at the public runner and validation boundaries."""
from pathlib import Path
import json
import shutil
import sys
import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'framework'),str(ROOT/'scripts'),str(ROOT/'revision_tools'),str(ROOT/'reproduction')]
from core import Step
from methods import BaseCDF,AdaptiveMemoryDBR
import run_revision as rr
import run_report_window_sensitivity as rw
import run_state_sensitivity as rs
import validate_report_windows as vw
from revision_integrity import (completed_payload_names,read_json,verify_complete,write_json)
from reference_integrity import verify_reference


def _steps(n=5,offset=0):
    return [Step(t=offset+i,u=float(u),y=float(u),F_t_inv=lambda p:p)
            for i,u in enumerate(np.linspace(.1,.9,n))]


@pytest.fixture
def tiny_runner(monkeypatch):
    def methods():
        return {'Base':BaseCDF(),'AMDBR-core-noidentity':AdaptiveMemoryDBR(
            lambdas=(.8,1.),bins=8,include_identity=False)}
    def synthetic(scenario,seed):
        return {'warmup':_steps(3),'pre_online':_steps(2,3),'online':_steps(5,5),
                'meta':{'scenario':scenario,'seed':seed,'n_online':5}}
    monkeypatch.setattr(rr,'methods_factory',methods)
    monkeypatch.setattr(rr,'synthetic_dataset',synthetic)
    return methods


def test_completed_runner_resume_checks_payload_hash_before_acceptance(tmp_path,tiny_runner):
    task=('synthetic','no_shift',6100,str(tmp_path/'run'),'unused')
    dest=Path(rr.run_cell(task))
    receipt=(dest/'COMPLETE.json').read_bytes()
    assert rr.run_cell(task)==str(dest)
    assert (dest/'COMPLETE.json').read_bytes()==receipt
    # Changing one prediction array must not be accepted even though both
    # its CSV and NPZ still exist and the root protocol has not changed.
    payload=dest/'Base.npz'
    with np.load(payload) as data:
        arrays={key:data[key] for key in data.files}
    arrays['lower'][0,0]+=.01
    np.savez_compressed(payload,**arrays)
    with pytest.raises(ValueError,match='payload hash mismatch'):
        rr.run_cell(task)


def test_partial_checkpoint_rejects_cross_seed_receipt(tmp_path,tiny_runner):
    paths=[]
    for seed in (6100,6101):
        paths.append(Path(rr.run_cell(('synthetic','no_shift',seed,str(tmp_path),'unused'))))
    first,second=paths
    (first/'COMPLETE.json').unlink()
    shutil.copy2(second/'Base_CHECKPOINT.json',first/'Base_CHECKPOINT.json')
    with pytest.raises(ValueError,match='identity mismatch'):
        rr.run_cell(('synthetic','no_shift',6100,str(tmp_path),'unused'))


def test_resume_rejects_missing_am_diagnostic_and_old_unreceipted_cell(tmp_path,tiny_runner):
    task=('synthetic','no_shift',6100,str(tmp_path/'run'),'unused')
    dest=Path(rr.run_cell(task))
    (dest/'AMDBR-core-noidentity_diagnostics.json').unlink()
    with pytest.raises(FileNotFoundError):rr.run_cell(task)
    stale=tmp_path/'stale'/'synthetic'/'no_shift_6100';stale.mkdir(parents=True)
    (stale/'Base.csv').write_text('method\nBase\n')
    with pytest.raises(ValueError,match='Unprovenanced or stale cell'):
        rr.run_cell(('synthetic','no_shift',6100,str(tmp_path/'stale'),'unused'))


def test_cell_config_identity_and_source_are_checked_against_current_code(tmp_path,tiny_runner,monkeypatch):
    task=('synthetic','no_shift',6100,str(tmp_path),'unused')
    dest=Path(rr.run_cell(task))
    expected=rr.expected_cell('synthetic','no_shift',6100)
    wrong=dict(expected,source_sha256='0'*64)
    with pytest.raises(ValueError,match='source/configuration/identity mismatch'):
        verify_complete(dest,wrong,completed_payload_names(tiny_runner()))
    monkeypatch.setattr(rr,'ALPHAS',(.05,.1,.2))
    with pytest.raises(ValueError,match='source/configuration/identity mismatch'):
        rr.run_cell(task)


@pytest.fixture
def tiny_window(tmp_path,tiny_runner,monkeypatch):
    # Exercise the real worker override and real validator with a tiny
    # in-memory stream; no live provider or mutable external input is used.
    specs=[{'label':'Seattle','name':'open_meteo_seattle_daily','max_eval':5,
            'train_frac':.2,'warm_frac':.45,'eval_start_frac':.7}]
    def load(name,**kwargs):
        f=kwargs['eval_start_frac'];start=int(20*f);end=min(20,start+5)
        return {'warmup':_steps(5,4),'pre_online':_steps(start-9,9),
                'online':_steps(end-start,start),'meta':{
                    'dataset':name,'n_total':20,'train_end':4,'warm_end':9,
                    'report_start':start,'eval_end':end,'n_online':end-start,
                    'n_warmup':5,'n_pre_online':start-9,'eval_start_frac':f,
                    'report_first_timestamp':'2020-01-16','report_last_timestamp':'2020-01-20'}}
    monkeypatch.setattr(rr,'DATASETS',specs)
    monkeypatch.setattr(rw,'DATASETS',specs)
    monkeypatch.setattr(rw,'methods_factory',tiny_runner)
    monkeypatch.setattr(rw,'FRACTIONS',(.75,))
    monkeypatch.setattr(rr,'load_dataset',load)
    out=tmp_path/'windows';out.mkdir()
    write_json(out/'protocol.json',rw.window_protocol())
    label,fraction,dest=rw.worker(('Seattle',.75,str(out),'unused'))
    assert specs[0]['eval_start_frac']==.7
    summary=pd.read_csv(Path(dest)/'summary.csv')
    summary.insert(0,'report_start_fraction',fraction)
    summary.to_csv(out/'summary.csv',index=False)
    return out,Path(dest)


def test_window_validator_accepts_matching_complete_dynamic_method_registry(tiny_window):
    out,dest=tiny_window
    result=vw.validate_windows(out)
    assert len(result['cells'])==1 and result['method_runs']==2
    assert result['payload_hashes_verified']


def test_window_validator_rejects_tampered_combined_summary(tiny_window):
    out,_=tiny_window
    frame=pd.read_csv(out/'summary.csv');frame.loc[0,'coverage']+=.1
    frame.to_csv(out/'summary.csv',index=False)
    with pytest.raises(AssertionError):vw.validate_windows(out)


def test_window_validator_rejects_missing_cell_and_stale_protocol(tiny_window):
    out,dest=tiny_window
    (dest/'COMPLETE.json').unlink()
    with pytest.raises(FileNotFoundError):vw.validate_windows(out)
    protocol=read_json(out/'protocol.json');protocol['config_sha256']='0'*64
    write_json(out/'protocol.json',protocol)
    with pytest.raises(ValueError,match='Report-window protocol'):
        vw.validate_windows(out)


def test_reference_hash_check_precedes_loading_pinned_code(tmp_path):
    ref=tmp_path/'reference'
    shutil.copytree(ROOT/'tests/reference',ref,ignore=shutil.ignore_patterns('__pycache__'))
    assert verify_reference(ref)['verification']=='passed before reference import'
    path=ref/'online_conformal'/'faci.py'
    path.write_bytes(path.read_bytes()+b'\nraise RuntimeError("modified upstream")\n')
    with pytest.raises(ValueError,match='reference hash mismatch: online_conformal/faci.py'):
        verify_reference(ref)


def test_state_cell_receipt_rejects_cross_scenario_and_changed_payload(tmp_path,monkeypatch):
    monkeypatch.setattr(rs,'synthetic_dataset',lambda scenario,seed:{
        'warmup':_steps(3),'pre_online':[],'online':_steps(5,3),
        'meta':{'scenario':scenario,'seed':seed,'n_online':5}})
    task=('no_shift',6100,str(tmp_path),rr.fingerprint(),rs.config_fingerprint())
    rs.cell(task)
    dest=tmp_path/'no_shift_6100'
    methods=list(rs.STATE_CONFIG['methods'])
    verify_complete(dest,rs.expected_state_cell('no_shift',6100),completed_payload_names(methods))
    with pytest.raises(ValueError,match='identity mismatch'):
        verify_complete(dest,rs.expected_state_cell('location',6100),completed_payload_names(methods))
    payload=dest/'DBR-Fixed-B987.csv'
    payload.write_bytes(payload.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='payload hash mismatch'):rs.cell(task)


def test_main_rejects_duplicate_tasks_before_creating_outputs(tmp_path,monkeypatch):
    monkeypatch.setattr(sys,'argv',['run_revision.py','--out',str(tmp_path/'run'),
        '--data-dir','unused','--seeds','6100','6100'])
    with pytest.raises(ValueError,match='Duplicate dataset/scenario/seed'):rr.main()
    assert not (tmp_path/'run').exists()
