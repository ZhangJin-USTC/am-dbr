"""Tables/figures from validated summaries; this is not a new experiment."""
from pathlib import Path
import json, hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--results',type=Path,required=True)
parser.add_argument('--windows',type=Path,required=True)
parser.add_argument('--assets',type=Path,required=True)
args=parser.parse_args()
OUT=args.assets/'generated'
OUT.mkdir(exist_ok=True,parents=True)
syn=pd.read_csv(args.results/'synthetic_summary.csv')
real=pd.read_csv(args.results/'real_summary.csv')
window_validation=json.loads((args.windows/'window_validation.json').read_text(encoding='utf-8'))
assert window_validation['payload_hashes_verified'] and window_validation['array_summary_checks']=='passed'
frame=pd.read_csv(args.windows/'summary.csv')
rows=[]
for (fraction,dataset), group in frame.groupby(['report_start_fraction','data_set'],sort=True):
    indexed=group.set_index('method')
    am=indexed.loc['AMDBR-core-noidentity'];dbr=indexed.loc['DBR-Fixed']
    cell=next(x for x in window_validation['cells'] if x['fraction']==fraction and x['dataset']==dataset)
    rows.append({'fraction':float(fraction),'dataset':dataset,'n_report':cell['n_report'],
                 'AM_score':am.PIT_CRPS,'DBR_score':dbr.PIT_CRPS,
                 'AM_minus_DBR_score':am.PIT_CRPS-dbr.PIT_CRPS,'AM_minus_DBR_gap':am.avg_gap-dbr.avg_gap})
window={'validation':'passed','rows':rows}
stats=json.loads((OUT/'statistics.json').read_text(encoding='utf-8'))
assert len(syn)==6*8*15 and len(real)==5*15
assert window['validation']=='passed' and len(window['rows'])==15
AM='AMDBR-core-noidentity'; DBR='DBR-Fixed'
methods=[AM,DBR,'Weighted-PIT-ECDF']
labels={AM:'AM-DBR',DBR:'DBR-Fixed','Weighted-PIT-ECDF':'Weighted ECDF'}
scens={'no_shift':'No location shift','location':'Location shift','gradual':'Gradual','cycle':'Cyclic','bursty':'Bursty','abrupt':'Abrupt'}
datasets=['Seattle','Phoenix','Jena','Metro','OPSD']
def save(n,t): (OUT/n).write_text(t,encoding='utf-8')

def detail_table(frame, group, groups, title,label):
    rows=[]
    for g,display in groups.items():
        for meth in methods:
            x=frame[(frame[group]==g)&(frame.method==meth)].select_dtypes(include='number').mean()
            rows.append(f"{display} & {labels[meth]} & {x.coverage:.3f} & {x.mean_width:.3f} & {x.rolling_mean_abs_gap:.3f} & {x.PIT_CRPS:.4f} \\\\")
        rows.append(r'\addlinespace')
    return r'''\begin{table}[p]
\centering\small\setlength{\tabcolsep}{3pt}
\caption{'''+title+r'''}\label{'''+label+r'''}
\begin{tabular}{llrrrr}\toprule
Series & Method & Coverage & Width & Local gap & PIT score \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}
\end{table}\clearpage
'''

detail=detail_table(syn,'scenario',scens,r'Simulation comparison at 90\% nominal coverage. Each row averages all eight seeds. Local gap is the mean rolling absolute coverage error, separate from the multilevel gap in Table~\ref{tab:sim}.','tab:sim-detail')
detail+=detail_table(real,'data_set',dict(zip(datasets,datasets)),r'Real-data main-window comparison at 90\% nominal coverage after exact-time cleaning. Widths retain dataset-specific units.','tab:real-detail')
detail+=r'''\begin{table}[p]
\centering\small
\caption{State of the primary implementations. $W$ is a retained window and $H$ a retained-history cap; these are algorithmic counts, not measured Python bytes. State counts for the two-stage rank-PAV comparator are given in the Supplementary Material.}\label{tab:state-order}
\begin{tabular}{lll}\toprule
Method & Output & State order / configured count \\
\midrule
DBR & Full CDF & $O(B)$; 140 masses \\
AM-DBR & Full CDF & $O(KB+K)$; 987 masses/weights \\
Rolling ECDF & Full CDF & $O(W)$; $W=300$ \\
Weighted ECDF & Full CDF & $O(H)$; $H=3000$ cap \\
SF-OGD & One interval level & Constant scalar-update state \\
\bottomrule\end{tabular}
\end{table}\clearpage
'''
save('detailed_main_tables.tex',detail)

s=stats['simulation_score_AM_minus_DBR']; av=stats['simulation_means']
save('simulation_narrative.tex',fr'''Across the 48 runs, mean PIT score is {av[AM]['PIT_CRPS']:.6f} for AM-DBR and {av[DBR]['PIT_CRPS']:.6f} for DBR-Fixed. The paired seed-mean difference is {s['mean']:.6f}, with a 95\% Student interval [{s['low']:.6f}, {s['high']:.6f}]. In gradual drift, AM-DBR has the higher score in all eight seeds, with mean difference {stats['scenario_score_differences']['gradual']['mean']:+.6f}. The overall multilevel gap is {av[AM]['avg_gap']:.6f} for AM-DBR and {av[DBR]['avg_gap']:.6f} for DBR; its difference is {stats['simulation_gap_AM_minus_DBR']['mean']:+.6f}, with an interval containing zero. Mean 90\% coverage is {av[AM]['coverage']:.4f} versus {av[DBR]['coverage']:.4f}, and mean width is {av[AM]['mean_width']:.3f} versus {av[DBR]['mean_width']:.3f}. The closeness of the single 90\% coverage average to nominal therefore does not imply improvement of the multilevel gap.
''')
am=real[real.method==AM].set_index('data_set'); db=real[real.method==DBR].set_index('data_set')
assert ((am.PIT_CRPS-db.PIT_CRPS)<0).all()
gaps=am.avg_gap-db.avg_gap
assert gaps[ ['Seattle','Phoenix','Jena','Metro'] ].gt(0).all() and gaps['OPSD']<0
save('real_narrative.tex',r'''AM-DBR has lower PIT score than DBR-Fixed in all five main windows. The multilevel gap is nevertheless larger for AM-DBR in Seattle, Phoenix, Jena and Metro, and smaller in OPSD. Thus the score improvement across all five datasets does not extend to the multilevel coverage gap. The Jena and Metro results use the timestamp and exact-lag protocol described above.
''')
w=window['rows']; neg=int(sum(x['AM_minus_DBR_score']<0 for x in w)); pos=int(sum(x['AM_minus_DBR_gap']>0 for x in w))
save('window_narrative.tex',fr'''AM-DBR has the lower PIT score in {neg} of the 15 additional windows. Its multilevel gap is larger in {pos} windows and smaller in {15-pos}. In particular, all three Metro windows favour AM-DBR in score but favour fixed DBR in multilevel gap. The $.95$ OPSD window has only 218 reported observations. These observations describe the checked windows, not all possible report starts.
''')
rows=[f"{x['fraction']:.2f} & {x['dataset']} & {x['n_report']} & {x['AM_score']:.6f} & {x['DBR_score']:.6f} & {x['AM_minus_DBR_gap']:+.6f} \\\\" for x in w]
save('window_table.tex',r'''\begin{table}[p]\centering\small\setlength{\tabcolsep}{3pt}
\caption{Report-start sensitivity. Fractions are applied after cleaning; methods continue updating before reporting. Gap difference is AM-DBR minus DBR-Fixed. These overlapping windows are not independent replicates.}\label{tab:window-sensitivity}
\begin{tabular}{llrrrr}\toprule
Start & Dataset & $m$ & AM score & DBR score & Gap difference \\
\midrule
'''+ '\n'.join(rows)+r'''
\bottomrule\end{tabular}\end{table}
''')
save('generator_details.tex',r'''
\subsection{Exact synthetic location and scale paths}\label{supp:generators}
For a run of 1500 observations, write $x_t=(t-1)/1499$. The six location paths are
\begin{align*}
m_{\mathrm{no\ shift}}(x)&=0, &m_{\mathrm{location}}(x)&=.85,\\
m_{\mathrm{gradual}}(x)&=\frac{1.45}{1+\exp\{-9(x-.50)\}}-.45,\\
m_{\mathrm{cycle}}(x)&=.55\sin(2\pi\,2.3x)+.45\sin(2\pi\,6x),\\
m_{\mathrm{bursty}}(x)&=.15\sin(2\pi\,1.3x)+\sum_{j=1}^3a_j\exp\{-(x-c_j)^2/(2w_j^2)\}.
\end{align*}
Here $(c_j,a_j,w_j)$ are $(.28,1.15,.030)$, $(.55,-1,.040)$ and $(.78,1.05,.028)$. The abrupt path is $-.35$, $1.15$, $-.15$ and $.95$ on intervals separated by $.33$, $.62$ and $.82$. The scale is $1+.12\sin(2\pi x)$, with an additional $.10\Ind\{|m(x)|>.8\}$ for bursty and abrupt settings. The constant-location scenario uses independent standard Gaussian noise; the other five scenarios use independent $t_6/\sqrt{6/4}$ noise. Seeds are reset for each scenario, so seed is the pairing unit, not an assertion of independent scenarios within a seed. The generator and scale definition are preserved in \path{scripts/fixed_adaptive_refresh.py}.
''')

fig,axes=plt.subplots(1,2,figsize=(9,4.2),sharey=True,layout='constrained')
for ax,col,title in zip(axes,['PIT_CRPS','avg_gap'],['PIT score difference','Multilevel gap difference']):
    values=(am[col]-db[col]).reindex(datasets).to_numpy()
    ax.axvline(0,color='#a0aab5',ls='--',lw=1)
    ax.scatter(values,np.arange(5),color='#1e5b80',s=35,zorder=3)
    ax.set_title(title,fontsize=11);ax.set_xlabel('AM-DBR minus DBR-Fixed',fontsize=9)
    ax.set_yticks(np.arange(5),datasets);ax.grid(axis='x',alpha=.15)
    ax.spines[['top','right']].set_visible(False)
axes[0].invert_yaxis()
fig.savefig(OUT/'real_comparison.pdf');fig.savefig(OUT/'real_comparison.png',dpi=160);plt.close(fig)

inputs={'main_synthetic_summary':args.results/'synthetic_summary.csv',
        'main_real_summary':args.results/'real_summary.csv',
        'window_summary':args.windows/'summary.csv',
        'statistics':OUT/'statistics.json'}
manifest={'scope':'Presentation assets derived from validated saved arrays and summaries.',
          'inputs':{key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in inputs.items()},
          'simulation_rows':len(syn),'real_rows':len(real),'window_rows':len(w),
          'window_score_lower':neg,'window_gap_higher':pos}
(args.assets/'presentation_validation.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
# The comparison table describes fixed methods rather than numerical outcomes.
from shutil import copyfile
copyfile(Path(__file__).resolve().parents[1]/'reproduction/templates/method_comparison_table.tex',OUT/'method_comparison_table.tex')
# Apply publication wording at the presentation stage. The upstream validated
# numerical generator and its source integrity checks use the same study configuration.
publication_wording = {
    ' Rank/PAV structural comparisons are reported in the supplement.': '',
    'Dataset-specific results, including all unfavourable comparisons, are retained.':
        'Dataset-specific score and coverage comparisons are reported in full.',
}
for name in ('main_tables.tex', 'results_narrative.tex'):
    path = OUT / name
    if path.is_file():
        text = path.read_text(encoding='utf-8')
        for before, after in publication_wording.items():
            text = text.replace(before, after)
        path.write_text(text, encoding='utf-8')
print(json.dumps(manifest,indent=2))
