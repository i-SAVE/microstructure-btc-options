"""Figures for the journal version (A5 text block 108 mm; 8 pt Liberation Sans; greyscale-safe; PNG 600 dpi + PDF)."""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = sys.argv[1]
D = os.path.join(ROOT, 'data')
OUT = os.path.join(ROOT, 'figs')
os.makedirs(OUT, exist_ok=True)
E = json.load(open(os.path.join(D, 'estimation_results.json')))
BT = json.load(open(os.path.join(D, 'backtest_results.json')))
X = json.load(open(os.path.join(D, 'extra_results.json')))
EM = json.load(open(os.path.join(D, 'estimation_mid_results.json')))
HQ = json.load(open(os.path.join(D, 'hedge_quote_iv.json')))
from matplotlib.ticker import FuncFormatter  # noqa: E402
COMMA = FuncFormatter(lambda v, pos: ('%g' % round(v, 6)).replace('.', ',').replace('-', '\u2212'))


def commas(ax, x=True, y=True):
    if x and ax.get_xscale() == 'linear':
        ax.xaxis.set_major_formatter(COMMA)
    if y and ax.get_yscale() == 'linear':
        ax.yaxis.set_major_formatter(COMMA)

W = 108 / 25.4
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial'], 'font.size': 8,
                     'axes.linewidth': 0.6, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
                     'legend.fontsize': 7, 'lines.linewidth': 1.0, 'axes.grid': False, 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6})
GREYS = ['#000000', '#555555', '#888888', '#222222', '#666666', '#999999']


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=600, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 1 system scheme
def fig1():
    """Scheme of the system (separate script fig_scheme.py; also writes Fig1.svg). rc_context keeps its font
    settings from leaking into Figs. 2-4."""
    import runpy
    argv = sys.argv
    sys.argv = [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig_scheme.py'), OUT]
    with plt.rc_context():
        runpy.run_path(sys.argv[0], run_name='__main__')
    sys.argv = argv


# ---------------------------------------------------------------- Fig. 2 estimation layer
def fig2():
    est = E['estimation']
    eq = EM['estimation']
    names = [('BS', 'Пост. волатильность'), ('PARAM', 'Параметрическая'), ('GB_GEO', 'Бустинг, $x^{G}$'),
             ('RF_GEO', 'Лес, $x^{G}$'), ('RF_MICRO', 'Лес, $x^{G},x^{L},x^{M}$'), ('RF_LEAK', 'Лес с утечкой*')]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.3), gridspec_kw={'width_ratios': [1.2, 1]})
    y = np.arange(len(names))[::-1]
    for yi, (k, lab) in zip(y, names):
        for src, off, mk, fc in [(est, 0.13, 'o', 'black'), (eq, -0.13, 's', 'white')]:
            v = src[k]
            lo, hi = v['ci_test']
            a1.errorbar(v['rmse_test'], yi + off, xerr=[[v['rmse_test'] - lo], [hi - v['rmse_test']]], fmt=mk,
                        mfc=fc, mec='black', ecolor='black', ms=3.3, elinewidth=0.7, capsize=1.6, linestyle='none')
    a1.plot([], [], 'o', color='black', ms=3.3, label='маркир. IV')
    a1.plot([], [], 's', mfc='white', mec='black', ms=3.3, label='IV котир.')
    a1.legend(frameon=False, loc='lower right', fontsize=6.3, handletextpad=0.2, borderaxespad=0.0, bbox_to_anchor=(1.03, 0.04))
    a1.set_yticks(y)
    a1.set_yticklabels([n[1] for n in names], fontsize=6.6)
    a1.set_xlabel('RMSE(IV), тест')
    a1.set_xlim(0.0, 0.52)
    a1.text(-0.02, 1.02, 'а)', transform=a1.transAxes, fontsize=8, ha='right', va='bottom')
    q = np.arange(1, 5)
    rm, rq = E['regimes']['s_rel'], eq['by_spread_q']
    for vals, lab, mk, ls, fc in [(rm['RF_GEO'], 'маркир., лес $x^{G}$', '^', ':', 'white'),
                                  (rm['RF_MICRO'], 'маркир., лес полн.', 'o', '-', 'black'),
                                  (rq['GB_GEO'], 'котир., бустинг $x^{G}$', 'v', '--', 'white'),
                                  (rq['RF_MICRO'], 'котир., лес полн.', 's', '-.', 'black')]:
        a2.plot(q, vals, marker=mk, linestyle=ls, color='black', ms=3.3, mfc=fc, label=lab)
    a2.set_xticks(q)
    a2.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    a2.set_xlabel('квартиль относительного спреда')
    a2.set_ylabel('RMSE(IV), тест')
    a2.set_ylim(0, 0.52)
    a2.legend(frameon=False, loc='upper left', fontsize=5.9, handlelength=2.6, borderaxespad=0.05, labelspacing=0.3)
    a2.text(-0.02, 1.02, 'б)', transform=a2.transAxes, fontsize=8, ha='right', va='bottom')
    for a in (a1, a2):
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
    commas(a1, x=True, y=False)
    commas(a2, x=False, y=True)
    fig.tight_layout(w_pad=0.8)
    save(fig, 'Fig2')


# ---------------------------------------------------------------- Fig. 3 attenuation
def fig3():
    est, eq = E['estimation'], EM['estimation']
    hed = X['hourly_models']
    mk = {'BS': ('s', 'Пост. волатильность'), 'PARAM': ('D', 'Параметрическая'), 'GB_GEO': ('v', 'Бустинг, $x^{G}$'),
          'RF_GEO': ('^', 'Лес, $x^{G}$'), 'RF_MICRO': ('o', 'Лес, $x^{G},x^{L},x^{M}$')}
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.15), sharey=False)
    baseM = hed['BS']['rmse']
    baseQ = HQ['BS']['rmse']
    for k, (m, lab) in mk.items():
        x = est[k]['rmse_test']
        g = hed[k]['gain_vs_BS']
        lo, hi = hed[k]['ci']
        a1.errorbar(x, 100 * g / baseM, yerr=[[100 * (g - lo) / baseM], [100 * (hi - g) / baseM]], fmt=m, color='black',
                    mfc='white' if k in ('PARAM', 'GB_GEO') else 'black', ms=3.6, elinewidth=0.7, capsize=1.6, label=lab)
        xq = eq[k]['rmse_test']
        gq = baseQ - HQ[k]['rmse']
        loq, hiq = HQ[k]['gain_vs_BS_ci']
        a2.errorbar(xq, 100 * gq / baseQ, yerr=[[100 * (gq - loq) / baseQ], [100 * (hiq - gq) / baseQ]], fmt=m,
                    color='black', mfc='white' if k in ('PARAM', 'GB_GEO') else 'black', ms=3.6, elinewidth=0.7, capsize=1.6)
    for a, t, xl in [(a1, 'а)', 'RMSE маркировочной IV'), (a2, 'б)', 'RMSE IV котировки')]:
        a.axhline(0, color='black', lw=0.5, ls=':')
        a.invert_xaxis()
        a.set_xlabel(xl + ', тест')
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)
        a.text(-0.02, 1.02, t, transform=a.transAxes, fontsize=8, ha='right', va='bottom')
        commas(a)
    a1.set_ylabel('снижение RMSE результата\nхеджирования, %')
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, fontsize=6.4, bbox_to_anchor=(0.5, -0.13),
               handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(w_pad=1.2)
    save(fig, 'Fig3')


# ---------------------------------------------------------------- Fig. 4 Pareto frontier
def fig4():
    F = BT['frontier']
    P = BT['policies']['2.5']
    fig, ax = plt.subplots(figsize=(W, 2.3))
    sty = {'threshold': ('Фиксированный порог', 'o', '-', 'white'), 'band': ('Фиксированная полоса', 's', '--', 'black'),
           'ww': ('Полоса Уолли–Уилмотта', '^', ':', 'black'), 'adaptive': ('Адаптивный порог', 'D', '-.', 'white')}
    sel_ad = P['adaptive']['params']
    for fam, (lab, m, ls, fc) in sty.items():
        rows = F[fam]
        if fam == 'adaptive':
            rows = [r for r in rows if (r['params']['a'], r['params']['b'], r['params']['c']) == (sel_ad['a'], sel_ad['b'], sel_ad['c'])]
        c = np.array([r['test']['cost'] for r in rows])
        e = np.array([r['test']['rmse'] for r in rows])
        o = np.argsort(c)
        ax.plot(c[o], e[o], marker=m, linestyle=ls, color='black', mfc=fc, ms=3.2, lw=0.9, label=lab)
        st = P[fam]['test']
        ax.plot(st['cost'], st['rmse'], marker=m, color='black', mfc=fc, ms=6.5, mew=1.3, linestyle='none')
    ab = X['adaptive_band_frontier']
    c = np.array([r['test']['cost'] for r in ab])
    e = np.array([r['test']['rmse'] for r in ab])
    ax.plot(c, e, marker='v', linestyle=(0, (1, 2.5)), color='#555555', mfc='white', ms=3.0, lw=0.8, label='Адаптивная полоса*')
    for fam, (lab, m) in {'hourly': ('Почасовая', 'X'), 'leland': ('Лиланд', 'P')}.items():
        st = P[fam]['test']
        ax.plot(st['cost'], st['rmse'], marker=m, color='black', ms=6.5, linestyle='none', label=lab)
    ax.set_xscale('log')
    ax.set_xlim(70, 2500)
    ax.set_ylim(69.6, 73.2)
    ax.set_xlabel('суммарные издержки на тесте, USDT (логарифмическая шкала)')
    ax.set_ylabel('RMSE пошагового P&L, USDT')
    ax.legend(frameon=False, loc='upper center', ncol=3, columnspacing=0.9, handlelength=3.2, fontsize=6.4,
              bbox_to_anchor=(0.5, -0.24))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([100, 200, 500, 1000, 2000])
    ax.set_xticklabels(['100', '200', '500', '1000', '2000'])
    commas(ax, x=False)
    save(fig, 'Fig4')


fig1()
fig2()
fig3()
fig4()
print('figures written to', OUT)
