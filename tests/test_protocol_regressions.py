"""Regression tests for preprocessing, scores, intervals and online updates."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import norm
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'framework'), str(ROOT/'real_data')]
from core import Step
from methods import BaseCDF, StaticPIT, OCC, DirectBinnedRecalibration, AdaptiveMemoryDBR, RollingPITECDF, WeightedPITECDF, DCP
from loaders import _autoregressive_frame, verify_pinned_input
from experiment import _normal_step
from cdf_tools import inverse_cdf
from residual_baselines import ResidualSAOCP
from residual_baselines import ResidualACI, _inverted_cdf_quantile


def test_unsorted_duplicate_hours_cannot_enter_own_lag():
    dt=pd.to_datetime(['2020-01-01 02:00','2020-01-01 00:00','2020-01-01 01:00','2020-01-01 01:00','2020-01-01 03:00'])
    times,X,y=_autoregressive_frame(dt,[20,0,10,10,30],(1,),frequency='h')
    assert times.is_monotonic_increasing and times.is_unique
    np.testing.assert_array_equal(y,[10,20,30])
    np.testing.assert_array_equal(X.y_lag_1,[0,10,20])
    assert X.attrs['preprocessing']['raw_duplicate_rows']==1
    assert X.attrs['preprocessing']['retained_nonunit_gaps']==0


def test_missing_hour_excludes_next_forecast_instead_of_row_shift():
    dt=pd.to_datetime(['2020-01-01 00:00','2020-01-01 01:00','2020-01-01 03:00','2020-01-01 04:00'])
    times,X,y=_autoregressive_frame(dt,[0,10,30,40],(1,),frequency='h')
    assert list(times.dt.hour)==[1,4]
    np.testing.assert_array_equal(X.y_lag_1,[0,30])
    assert X.attrs['preprocessing']['retained_nonunit_gaps']==1


def test_future_append_does_not_change_historical_features():
    dt=pd.date_range('2020-01-01', periods=40)
    t1,x1,y1=_autoregressive_frame(dt[:20],np.arange(20.),(1,7))
    t2,x2,y2=_autoregressive_frame(dt,np.arange(40.),(1,7))
    pd.testing.assert_frame_equal(x1,x2.iloc[:len(x1)])
    np.testing.assert_array_equal(y1,y2[:len(y1)])


def test_raw_input_replacement_rejected(tmp_path):
    p=tmp_path/'opsd_germany_daily.csv';p.write_bytes(b'changed source')
    with pytest.raises(ValueError,match='hash mismatch'):
        verify_pinned_input(p)


@pytest.mark.parametrize('factory',[BaseCDF,StaticPIT,lambda:OCC(bins=12),lambda:DirectBinnedRecalibration(bins=12),lambda:AdaptiveMemoryDBR(bins=12),RollingPITECDF,WeightedPITECDF])
def test_exact_score_agrees_with_independent_quadrature(factory):
    m=factory();m.warmup([Step(t=i,u=u) for i,u in enumerate([.02,.2,.2,.5,.71,.94])])
    x,_,_=m._pit_knots()
    for u in [0.,.20347,.88,1.]:
        truth=quad(lambda z: (float(m.predict_g(np.array([z]))[0])-float(z>=u))**2,0,1,points=sorted(set([*x,u])),epsabs=1e-11)[0]
        assert m.score_pit(u)==pytest.approx(truth,abs=2e-12)


def test_static_anchors_ties_and_inverse_flat():
    m=StaticPIT();m.warmup([Step(i,u) for i,u in enumerate([.2,.2,.8])])
    assert list(m.predict_g(np.array([0.,.2,1.])))==[0.,2/3,1.]
    assert inverse_cdf([0,.2,.7,1],[0,.5,.5,1],.5)==.2
    lo,hi=m.predict_interval(lambda p:p,.1)
    np.testing.assert_allclose(m.predict_g(np.array([lo,hi])),[.05,.95])


def test_unbounded_dcp_small_sample_and_no_gaussian_tail_truncation():
    m=DCP();m.update(Step(0,.51))
    lo,hi=m.predict_interval(norm.ppf,.05)
    assert np.isneginf(lo) and np.isposinf(hi)
    step=_normal_step(1,8.,0.,1.)
    assert step.u > 1-1e-6
    assert step.F_t_inv(1-1e-10)==pytest.approx(norm.ppf(1-1e-10))


def test_saocp_zero_warmup_is_defined():
    m=ResidualSAOCP();m.warmup([Step(i,.5,y=0.,F_t_inv=lambda p:0.) for i in range(10)])
    m.update(Step(11,.5,y=1.,F_t_inv=lambda p:0.))
    assert np.isfinite(m._radius())


@pytest.mark.parametrize('alpha',[.05,.1,.2,.3,.5])
def test_aci_order_statistic_optimization_preserves_intervals(alpha):
    scores=np.r_[np.zeros(20),np.ones(20),np.abs(np.random.default_rng(913).standard_t(4,400))]
    steps=[Step(i,.5,y=float(s),F_t_inv=lambda p:0.) for i,s in enumerate(scores)]
    m=ResidualACI(alpha_target=alpha);m.warmup(steps[:30])
    history=list(scores[:30]);level=alpha
    for step in steps[30:]:
        expected=_inverted_cdf_quantile(history,1-level)
        lo,hi=m.predict_interval(lambda p:0.,alpha)
        assert hi==expected and lo==-expected
        level=float(np.clip(level+.005*(alpha-float(step.y>expected)),0,1))
        history.append(step.y);m.update(step)
        assert m.alpha_t==level
