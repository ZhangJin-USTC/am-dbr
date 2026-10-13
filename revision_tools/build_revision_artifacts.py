"""Validate every completed cell and regenerate all current manuscript assets.

The CSV/NPZ files, rather than cached TeX or hand-entered numbers, are the source
for tables, figures and numerical conclusions. Missing cells are fatal.
"""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'framework'),str(ROOT/'scripts')]
from run_revision import SCENARIOS,fingerprint,expected_cell,main_protocol
from run_state_sensitivity import expected_state_cell,state_protocol
from revision_integrity import completed_payload_names,read_json,require_equal,verify_complete
from adaptive_memory_realdata_study import methods_factory,DATASETS,ALPHAS

AM='AMDBR-core-noidentity';DBR='DBR-Fixed'
ORDER=list(methods_factory())
# The full registry is still validated and written to the supplementary
# tables. The main paper presents rank/PAV only as a short structural ablation,
# so its detailed method rows are deliberately excluded here.
MAIN_EXCLUDED={'OCC-Fixed','OCC-Adaptive'}
LABELS={'Base':'Base','Static-PIT':'Static PIT','Rolling-PIT-ECDF':'Rolling ECDF','Weighted-PIT-ECDF':'Weighted ECDF','DCP-score':'DCP score',DBR:'DBR fixed','OCC-Fixed':'Rank/PAV fixed','OCC-Adaptive':'Rank/PAV adaptive',AM:'AM-DBR','ACI-residual-projected':'ACI projected','NExCP-residual-empirical':'NExCP empirical','PID-residual-projected':'PID projected','FACI-residual':'FACI','SF-OGD-residual':'SF-OGD','SAOCP-residual':'SAOCP'}
SCENARIO_NAMES={'no_shift':'No location shift','location':'Location shift','gradual':'Gradual','cycle':'Cyclic','bursty':'Bursty','abrupt':'Abrupt'}

def check_close(actual,expected,label):
    if not np.isclose(actual,expected,rtol=1e-11,atol=1e-12,equal_nan=True):
        raise AssertionError(f'{label}: {actual} != {expected}')

def validate_cell(dest,methods):
    if not (dest/'summary.csv').is_file():raise FileNotFoundError(dest/'summary.csv')
    summary=pd.read_csv(dest/'summary.csv')
    if summary.method.duplicated().any() or set(summary.method)!=set(methods):raise AssertionError(f'Method schema: {dest}')
    reference_y=None
    for name in methods:
        row=summary.set_index('method').loc[name]
        with np.load(dest/f'{name}.npz',allow_pickle=False) as a:
            y=a['y'];n=len(y);alpha=a['alphas'];lo=a['lower'];hi=a['upper']
            np.testing.assert_array_equal(alpha,ALPHAS)
            if lo.shape!=(5,n) or hi.shape!=(5,n) or not np.isfinite(y).all() or np.isnan(lo).any() or np.isnan(hi).any() or np.any(lo>hi):raise AssertionError(f'Invalid intervals: {dest}/{name}')
            if reference_y is None:reference_y=y.copy()
            else:np.testing.assert_array_equal(reference_y,y)
            cov=((lo<=y)&(y<=hi)).mean(axis=1);gap=np.abs(cov-(1-alpha))
            for i,v in enumerate(alpha):check_close(gap[i],row[f'err@alpha={v:g}'],f'{dest}/{name}/{v}')
            check_close(cov[1],row.coverage,'coverage');check_close(gap.mean(),row.avg_gap,'avg_gap');check_close(gap.max(),row.max_gap,'max_gap')
            check_close(np.mean(hi[1]-lo[1]),row.mean_width,'width');check_close(np.median(hi[1]-lo[1]),row.median_width,'median width')
            if 'pit_crps' in a:
                if a['pit_crps'].shape!=(n,) or not np.isfinite(a['pit_crps']).all() or (a['pit_crps'] < -1e-12).any() or (a['pit_crps']>1+1e-12).any():raise AssertionError('Invalid score')
                check_close(a['pit_crps'].mean(),row.PIT_CRPS,'score')
                if a['calibrated_pit'].shape!=(n,) or not np.isfinite(a['calibrated_pit']).all() or np.any(a['calibrated_pit'] < 0) or np.any(a['calibrated_pit'] > 1):raise AssertionError('Invalid recalibrated PIT')
                if name==AM:
                    diagnostic=json.loads((dest/f'{name}_diagnostics.json').read_text())
                    if diagnostic.get('schema_version')!=2:
                        raise AssertionError('AM diagnostics must distinguish full history and report window')
                    report=diagnostic['report_window'];full=diagnostic['full_history']
                    if report['horizon']!=n or full['horizon']!=report['prior_updates']+n:
                        raise AssertionError('AM diagnostic horizon mismatch')
                    check_close(report['cumulative_aggregate_loss_exact'],a['pit_crps'].sum(),'AM report loss')
                    w=np.asarray(report['initial_expert_weights'])
                    if not np.isfinite(w).all() or np.any(w<0) or len(w)!=len(report['cumulative_expert_losses']):
                        raise AssertionError('Invalid AM report-start weights')
                    check_close(w.sum(),1.,'AM report-start weight sum')
            elif not pd.isna(row.PIT_CRPS):raise AssertionError('Score assigned to interval-only method')
    return summary,n

def paired(df,col,left=AM,right=DBR,by_scenario=False):
    index=['scenario','seed'] if by_scenario else ['seed']
    d=df.groupby(index+['method'])[col].mean().unstack('method')
    delta=(d[left]-d[right]).dropna().to_numpy();n=len(delta);m=delta.mean();radius=student_t.ppf(.975,n-1)*delta.std(ddof=1)/np.sqrt(n)
    return {'mean':float(m),'low':float(m-radius),'high':float(m+radius),'n':n,'positive':int((delta>0).sum())}

def fmt(v,d=4):return '--' if pd.isna(v) else f'{v:.{d}f}'
def table(caption,label,headers,rows,align=None):
    align=align or 'l'+'r'*(len(headers)-1)
    body='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)
    return '\\begin{table}[p]\n\\centering\n\\caption{'+caption+'}\\label{'+label+'}\n\\begin{tabular}{'+align+'}\n\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\n'+body+'\n\\bottomrule\n\\end{tabular}\n\\end{table}\n\\clearpage\n'

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--state',type=Path,required=True);p.add_argument('--manuscript',type=Path,required=True);args=p.parse_args()
    protocol=json.loads((args.results/'protocol.json').read_text())
    require_equal(protocol,main_protocol(list(range(6100,6108)),list(SCENARIOS),[s['label'] for s in DATASETS]),'Main protocol')
    synthetic=[];real=[];count=0;predictions=0
    for s in SCENARIOS:
        for seed in range(6100,6108):
            dest=args.results/'synthetic'/f'{s}_{seed}'
            verify_complete(dest,expected_cell('synthetic',s,seed),completed_payload_names(ORDER))
            d,n=validate_cell(dest,ORDER);synthetic.append(d);count+=1;predictions+=n*len(ORDER)*5
            if set(d.scenario)!={s} or set(d.seed)!={seed}:raise ValueError(f'Synthetic identity mismatch: {dest}')
    for spec in DATASETS:
        dest=args.results/'real'/spec['label']
        verify_complete(dest,expected_cell('real',spec['label'],-1),completed_payload_names(ORDER))
        d,n=validate_cell(dest,ORDER);real.append(d);count+=1;predictions+=n*len(ORDER)*5
        if set(d.data_set)!={spec['label']} or set(d.seed)!={-1}:raise ValueError(f'Real identity mismatch: {dest}')
    sensitivity=[]
    state_protocol_path = args.state/'protocol.json'
    if not state_protocol_path.exists():
        raise FileNotFoundError(state_protocol_path)
    recorded_state_protocol = json.loads(state_protocol_path.read_text())
    require_equal(recorded_state_protocol,state_protocol(),'State protocol')
    for s in SCENARIOS:
        for seed in range(6100,6108):
            dest=args.state/f'{s}_{seed}'
            state_methods=['OCC-Fixed-B70','DBR-Fixed-B987']
            verify_complete(dest,expected_state_cell(s,seed),completed_payload_names(state_methods))
            d,n=validate_cell(dest,state_methods);sensitivity.append(d)
            if set(d.scenario)!={s} or set(d.seed)!={seed}:raise ValueError(f'State identity mismatch: {dest}')
    syn=pd.concat(synthetic,ignore_index=True);real=pd.concat(real,ignore_index=True);state=pd.concat(sensitivity,ignore_index=True)
    syn.to_csv(args.results/'synthetic_summary.csv',index=False);real.to_csv(args.results/'real_summary.csv',index=False)
    gen=args.manuscript/'generated';gen.mkdir(parents=True,exist_ok=True)
    stats={'simulation_score_AM_minus_DBR':paired(syn,'PIT_CRPS'),'simulation_gap_AM_minus_DBR':paired(syn,'avg_gap')}
    stats['scenario_score_differences']={s:paired(syn[syn.scenario==s],'PIT_CRPS') for s in SCENARIOS}
    overall=syn.groupby('method')[['PIT_CRPS','avg_gap','mean_width','coverage']].mean()
    realmean=real.groupby('method')[['PIT_CRPS','avg_gap']].mean()
    stats['simulation_means']=overall.to_dict('index');stats['real_means']=realmean.to_dict('index')
    ci=stats['simulation_score_AM_minus_DBR'];grad=stats['scenario_score_differences']['gradual']
    winners=int((real.pivot(index='data_set',columns='method',values='PIT_CRPS')[AM]<real.pivot(index='data_set',columns='method',values='PIT_CRPS')[DBR]).sum())
    narrative=f'''Across the 48 simulation runs, mean PIT-scale score is {overall.loc[AM,'PIT_CRPS']:.6f} for AM-DBR and {overall.loc[DBR,'PIT_CRPS']:.6f} for DBR-Fixed. Their paired difference, averaging the six scenarios within each seed, is {ci['mean']:.6f}, with a 95\\% Student interval [{ci['low']:.6f}, {ci['high']:.6f}] based on eight seeds. This interval describes replicate variability for these prespecified scenarios and is not a general population guarantee. In gradual drift the mean score difference is {grad['mean']:+.6f}; AM-DBR has the higher score in {grad['positive']} of eight seeds. Its overall multilevel gap is {overall.loc[AM,'avg_gap']:.6f}, versus {overall.loc[DBR,'avg_gap']:.6f} for DBR-Fixed. Thus the score improvement does not give a coverage-gap improvement. In the five real datasets AM-DBR has a lower PIT score than DBR-Fixed in {winners} datasets. Dataset-specific results, including all unfavourable comparisons, are retained.\n'''
    (gen/'results_narrative.tex').write_text(narrative,encoding='utf-8')
    full=[m for m in ORDER if m not in MAIN_EXCLUDED and pd.notna(overall.loc[m,'PIT_CRPS'])]
    rows=[[LABELS[m],fmt(overall.loc[m,'PIT_CRPS']),fmt(overall.loc[m,'avg_gap']),fmt(overall.loc[m,'mean_width'],3)] for m in full]
    main=table('Simulation means for the six primary full-distribution methods over six scenarios and eight seeds. Gap is the mean absolute coverage error over five levels; width is for 90\\% intervals. The historical rank/PAV structural ablation is reported in the supplement.','tab:sim',['Method','PIT score','Gap','Width'],rows)
    rows=[]
    for spec in DATASETS:
        r=real[real.data_set==spec['label']].set_index('method');rows.append([spec['label'],fmt(r.loc[AM,'PIT_CRPS']),fmt(r.loc[DBR,'PIT_CRPS']),fmt(r.loc[AM,'avg_gap']),fmt(r.loc[DBR,'avg_gap'])])
    main+=table('Dataset-specific comparison after chronological cleaning. Full comparator results are in the supplement.','tab:real',['Dataset','AM score','DBR score','AM gap','DBR gap'],rows)
    (gen/'main_tables.tex').write_text(main,encoding='utf-8')
    # Figures: paired differences, preserving the adverse gradual-drift result.
    fig,axes=plt.subplots(1,2,figsize=(9.0,4.3),sharey=True)
    for ax,col,title in zip(axes,['PIT_CRPS','avg_gap'],['(A) PIT score difference','(B) Multilevel gap difference']):
        vals=[paired(syn[syn.scenario==s],col) for s in SCENARIOS]
        means=np.array([v['mean'] for v in vals]);lo=np.array([v['low'] for v in vals]);hi=np.array([v['high'] for v in vals])
        ax.errorbar(means,np.arange(6),xerr=np.array([means-lo,hi-means]),fmt='o',color='#186f8c',ecolor='#52717d',capsize=3)
        ax.axvline(0,color='#777777',lw=.8);ax.set_title(title,fontsize=11);ax.set_xlabel('AM-DBR minus DBR-Fixed',fontsize=10);ax.grid(axis='x',alpha=.2)
        ax.xaxis.set_major_locator(MaxNLocator(4))
    axes[0].set_yticks(np.arange(6),[SCENARIO_NAMES[s] for s in SCENARIOS]);axes[0].invert_yaxis();fig.tight_layout()
    fig.savefig(gen/'score_coverage.pdf');fig.savefig(gen/'score_coverage.png',dpi=180);plt.close(fig)
    # Full all-method tables at all five levels, one scenario/dataset per page.
    supp=''
    for key,df in [('scenario',syn),('data_set',real)]:
        groups=list(SCENARIOS) if key=='scenario' else [s['label'] for s in DATASETS]
        for label in groups:
            d=df[df[key]==label].groupby('method').mean(numeric_only=True)
            rows=[[LABELS[m],fmt(d.loc[m,'PIT_CRPS']),fmt(d.loc[m,'mean_width'],2)]+[fmt(d.loc[m,f'err@alpha={a:g}'],3) for a in ALPHAS] for m in ORDER]
            title=SCENARIO_NAMES[label] if key=='scenario' else label
            supp+=table(f'{title}: all methods. $e_{{\\alpha}}$ is absolute coverage error at miscoverage $\\alpha$. Simulation entries average eight seeds; real-data entries use one chronological stream. Scores are undefined for interval-only methods.',f'tab:{key}-{label}',['Method','Score','Width','$e_{.05}$','$e_{.10}$','$e_{.20}$','$e_{.30}$','$e_{.50}$'],rows)
    clean=[];windows=[];metadata={}
    for spec in DATASETS:
        meta=json.loads((args.results/'real'/spec['label']/'metadata.json').read_text());metadata[spec['label']]=meta;q=meta['preprocessing']
        clean.append([spec['label'],q['raw_rows'],q['raw_duplicate_rows'],q['retained_rows'],q['excluded_rows'],q['retained_nonunit_gaps']])
        windows.append([spec['label'],meta['n_online'],meta['report_first_timestamp'][:10],meta['report_last_timestamp'][:10]])
    supp+=table('Cleaning diagnostics. Duplicates are extra rows at an existing timestamp; excluded rows are unique timestamps missing a target or required lag. Gaps count non-unit intervals among retained rows.','tab:cleaning',['Dataset','Raw','Duplicates','Retained','Excluded','Gaps'],clean)
    supp+=table('Corrected report windows. Date endpoints are inclusive; detailed timestamps and source-clock conventions are stored in the metadata.','tab:windows',['Dataset','$n$','First date','Last date'],windows,'lrrr')
    joined=pd.concat([syn,state]);state_stats={}
    for left,right in [('OCC-Fixed-B70',DBR),(AM,'DBR-Fixed-B987')]:
        state_stats[left+' minus '+right]={c:paired(joined,c,left,right) for c in ['PIT_CRPS','avg_gap']}
    stats['state_sensitivity']=state_stats
    a=state_stats['OCC-Fixed-B70 minus '+DBR];b=state_stats[AM+' minus DBR-Fixed-B987']
    (gen/'state_narrative.tex').write_text(f'''At 140 accumulators, fixed rank/PAV with 70 bins minus DBR with 140 bins has mean score difference {a['PIT_CRPS']['mean']:+.6f} and gap difference {a['avg_gap']['mean']:+.6f}. At 987 accumulators, AM-DBR minus DBR with 987 bins has score difference {b['PIT_CRPS']['mean']:+.6f} and gap difference {b['avg_gap']['mean']:+.6f}. These are averages over all 48 simulation runs, with configurations fixed before this sensitivity was executed. Neither result establishes universal dominance or equal byte-level storage.\n''',encoding='utf-8')
    (gen/'supp_tables.tex').write_text(supp,encoding='utf-8')
    (gen/'statistics.json').write_text(json.dumps(stats,indent=2,allow_nan=True),encoding='utf-8')
    (gen/'data_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    check={'main_cells':count,'main_method_runs':len(syn)+len(real),'main_interval_predictions_across_levels':predictions,'state_method_runs':len(state),'levels':list(ALPHAS),'source_fingerprint_verified':True,'cell_payload_hashes_verified':True,'array_summary_checks':'passed','scope':'main and count-state only; report-window sensitivity requires validate_report_windows.py','mandatory_assets':['main_tables.tex','supp_tables.tex','results_narrative.tex','state_narrative.tex','score_coverage.pdf']}
    for f in check['mandatory_assets']:
        if not (gen/f).is_file() or not (gen/f).stat().st_size:raise FileNotFoundError(gen/f)
    (args.results/'artifact_validation.json').write_text(json.dumps(check,indent=2),encoding='utf-8');print(json.dumps(check,indent=2))

if __name__=='__main__':main()
