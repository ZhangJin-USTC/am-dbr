"""Compare actual upstream score recurrences with a minimal forecaster adapter.

The adapter only supplies coverage/horizon and initial residuals; it does not
replace the downloaded prediction, quantile or update implementations.
"""
from pathlib import Path
import sys,types,importlib.util,json
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];E=Path(sys.argv[1]) if len(sys.argv)>1 else R/'outputs/upstream_validation';E.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(R/'framework'))
sys.path.insert(0,str(R/'reproduction'))
from reference_integrity import verify_reference
from revision_integrity import file_digest,write_json
reference_evidence=verify_reference(R/'tests/reference')
reference_evidence['adapter_sha256']=file_digest(Path(__file__))
write_json(E/'upstream_source_validation.json',reference_evidence)
from core import Step
from residual_baselines import ResidualFACI,ResidualSFOGD,ResidualSAOCP,ResidualNExCP
package=types.ModuleType('online_conformal');package.__path__=[str(R/'tests/reference/online_conformal')];sys.modules['online_conformal']=package
import online_conformal.utils as utils
class ForecasterAdapter:
    def __init__(self,*args,horizon=1,coverage=.9,initial_scores=(),**kwargs):
        self.horizon=horizon;self.coverage=coverage;self.residuals=utils.Residuals(horizon)
        self.residuals.extend(1,list(initial_scores))
    def quantile(self,arr,q):return utils.quantile(arr,q)
base=types.ModuleType('online_conformal.base');base.BasePredictor=ForecasterAdapter;sys.modules[base.__name__]=base
enb=types.ModuleType('online_conformal.enbpi');enb.EnbMixIn=type('EnbMixIn',(),{});sys.modules[enb.__name__]=enb
import online_conformal.faci as faci
import online_conformal.ogd as ogd
import online_conformal.saocp as saocp
split=types.ModuleType('online_conformal.split_conformal');split.SplitConformal=ForecasterAdapter;sys.modules[split.__name__]=split
import online_conformal.nex_conformal as nex
rng=np.random.default_rng(91824);scores=np.r_[np.abs(rng.standard_t(4,300)),np.zeros(20),np.ones(20)]
scores[35:40]=3.;scores[60]=40.
warm=scores[:30];online=scores[30:]
steps=[Step(t=i,u=.5,y=float(s),F_t_inv=lambda p:0.) for i,s in enumerate(scores)]
rows=[]
for alpha in [.05,.1,.2,.3,.5]:
    for label,local_class,ref_class in [('FACI',ResidualFACI,faci.FACI),('SF-OGD',ResidualSFOGD,ogd.ScaleFreeOGD),('SAOCP',ResidualSAOCP,saocp.SAOCP),('NExCP',ResidualNExCP,nex.NExConformal)]:
        local=local_class(alpha_target=alpha);local.warmup(steps[:30])
        if label=='FACI':
            ref=ref_class(horizon=1,coverage=1-alpha)
            ref.update(pd.Series(warm),pd.Series(np.zeros(30)),1)
        else:ref=ref_class(horizon=1,coverage=1-alpha,initial_scores=warm)
        diffs=[];first=None
        for i,s in enumerate(online,30):
            radius_local=local.predict_interval(lambda p:0.,alpha)[1]
            radius_ref=ref.quantile(ref.residuals.get(1),1-alpha) if label=='NExCP' else ref.predict(1)[1]
            diffs.append(abs(radius_local-radius_ref))
            if first is None and diffs[-1]>1e-10:first={'time_index':i,'local':float(radius_local),'reference':float(radius_ref)}
            local.update(steps[i])
            if label=='NExCP':ref.residuals.extend(1,[s])
            else:ref.update(pd.Series([s]),pd.Series([0.]),1)
        rows.append({'method':label,'alpha':alpha,'warmup':30,'n_predictions':len(diffs),'max_abs_radius_diff':max(diffs),'allclose_1e-10':bool(max(diffs)<1e-10),'first_difference':json.dumps(first),'local_floor_threshold':int(np.floor(1/alpha)),'reference_floor_threshold':int(np.floor(1/(1-ref.coverage)))})
pd.DataFrame(rows).to_csv(E/'upstream_recurrence_comparison.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
alpha=.1;scale=float(max(warm)*np.sqrt(3));local=ResidualSAOCP(alpha_target=alpha,max_scale=scale);ref=saocp.SAOCP(horizon=1,coverage=1-alpha,max_scale=scale)
trace=[]
for i,s in enumerate(scores):
    local.update(steps[i]);ref.update(pd.Series([s]),pd.Series([0.]),1)
    dif=abs(local._radius()-ref.predict(1)[1])
    if dif>1e-12:
        trace.append({'time_index':i,'score':float(s),'local_radius':local._radius(),'reference_radius':ref.predict(1)[1],'local_weights':{str(k):v.weight for k,v in local.experts.items()},'ref_weights':{str(k):v.w for k,v in ref.experts[1].items()}})
        if len(trace)>=3:break
(E/'saocp_first_divergence.json').write_text(json.dumps(trace,indent=2),encoding='utf-8')

assert all(row["allclose_1e-10"] for row in rows), "Upstream recurrence mismatch"
