# -*- coding: utf-8 -*-
"""Fig. 1 — scheme of the hedging system (journal style: white background, greyscale, thin lines).
Online loop at hour t: input data -> estimation -> decision -> execution (eqs. 1-3);
offline evaluation on the test sample: estimation accuracy, step result (4), metrics J (5), value of information (6).
Every symbol is defined in Section 3 of the manuscript. Usage: python fig_scheme.py <OUT_DIR>"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else '.'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Liberation Sans', 'Arial'],
                     'mathtext.fontset': 'custom', 'mathtext.rm': 'Liberation Sans',
                     'mathtext.it': 'Liberation Sans:italic', 'mathtext.bf': 'Liberation Sans:bold',
                     'mathtext.fallback': 'stixsans', 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'svg.fonttype': 'path'})

WMM, HMM = 108.0, 80.0
LW = 0.6            # pt, box and arrow lines
FS_T, FS = 6.8, 6.1  # pt: titles / body text
K = '#000000'
G = '#5a5a5a'        # secondary text
PANEL = '#f3f3f3'    # offline panel background

fig = plt.figure(figsize=(WMM / 25.4, HMM / 25.4))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, WMM)
ax.set_ylim(0, HMM)
ax.axis('off')


def box(x0, x1, y0, y1, title, lines, fc='white', ls='-', top_pad=1.3, step=2.95, title_gap=3.4, align='left'):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, lw=LW, ec=K, fc=fc, ls=ls, zorder=2))
    ax.text((x0 + x1) / 2, y1 - top_pad, title, ha='center', va='top', fontsize=FS_T, fontweight='bold', zorder=3)
    y = y1 - top_pad - title_gap
    for ln in lines:
        if isinstance(ln, tuple):
            txt, kw = ln
        else:
            txt, kw = ln, {}
        xa = x0 + 1.2 if align == 'left' else (x0 + x1) / 2
        ax.text(xa, y, txt, ha='left' if align == 'left' else 'center', va='top',
                fontsize=kw.get('fs', FS), color=kw.get('c', K), zorder=3)
        y -= kw.get('step', step)


def arrow(pts, ls='-', lw=LW, head=True):
    """polyline with an arrow head at the last point"""
    for (xa, ya), (xb, yb) in zip(pts[:-2], pts[1:-1]):
        ax.plot([xa, xb], [ya, yb], color=K, lw=lw, ls=ls, solid_capstyle='butt', zorder=4)
    (xa, ya), (xb, yb) = pts[-2], pts[-1]
    ax.add_patch(FancyArrowPatch((xa, ya), (xb, yb), arrowstyle='-|>' if head else '-', mutation_scale=5.5,
                                 lw=lw, color=K, ls=ls, shrinkA=0, shrinkB=0, zorder=4))


def label(x, y, s, ha='center', va='bottom', fs=FS, c=K, bg=False):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=c, zorder=5,
            bbox=dict(fc='white', ec='none', pad=0.4) if bg else None)


# ------------------------------------------------------------------ online loop (hour t)
YT, YB = 74.0, 50.0                  # top / bottom of the estimation, decision and execution boxes
B1 = (0.4, 27.4, 43.0, YT)           # input data
B2 = (32.0, 53.4)                    # estimation
B3 = (58.0, 83.4)                    # decision
B4 = (88.0, 107.6)                   # execution
label(0.4, 77.2, 'Онлайн-контур: шаг $t$ = 1 ч', ha='left', va='bottom', fs=FS, c=G)

box(B1[0], B1[1], B1[2], B1[3], 'Входные данные', [
    ('состояние $x_t=(x_t^{G},x_t^{L},x_t^{M})$', {'step': 3.6}),
    '$x_t^{G}$: $k_t$, $T_t$, тип $\\iota$',
    '$x_t^{L}$: $s_t^{rel}$, объём торгов, OI,',
    ('     объёмы bid/ask, лаги', {'step': 3.3}),
    ('$x_t^{M}$: $RV_t$', {'step': 3.6}),
    ('$S_t$, $c_t$ – цена и ставка', {'c': G}),
    ('издержек спота', {'c': G}),
])
box(B2[0], B2[1], YB, YT, 'Оценивание', [
    ('$\\hat{\\sigma}_t=f(x_t^{E})$', {'step': 3.4}),
    ('$\\Delta_t$, $\\Gamma_t$: формула', {'step': 2.8}),
    ('Блэка – Шоулза', {'step': 2.8}),
    ('при $r=0$; (2)', {'step': 3.4}),
    ('$f$: 5 моделей, п. 4.1', {'c': G}),
])
box(B3[0], B3[1], YB, YT, 'Решение', [
    ('$\\varphi_t=\\pi(\\hat{\\sigma}_t,\\Delta_t,\\Gamma_t,$', {'step': 2.9}),
    ('       $\\varphi_{t-1},x_t^{D})$,', {'step': 2.9}),
    ('$u_t=\\varphi_t-\\varphi_{t-1}$;  (3)', {'step': 3.2}),
    ('почасовое, Лиланд,', {'c': G, 'step': 2.6}),
    ('порог, полоса,', {'c': G, 'step': 2.6}),
    ('Уолли – Уилмотт,', {'c': G, 'step': 2.6}),
    ('адаптивный порог', {'c': G}),
])
box(B4[0], B4[1], YB, YT, 'Исполнение', [
    ('сделка $u_t$', {'step': 2.8}),
    ('в споте по цене $S_t$;', {'step': 3.4}),
    ('издержки', {'step': 2.9}),
    '$c_t\\,|u_t|\\,S_t$',
])

# data -> estimation (x^E) and data -> decision (x^D, bus under the estimation box)
yE = 64.0
arrow([(B1[1], yE), (B2[0], yE)])
label((B1[1] + B2[0]) / 2, yE + 0.5, '$x_t^{E}$')
yD, xD = 46.6, 64.0
arrow([(B1[1], yD), (xD, yD), (xD, YB)])
label(33.6, yD + 0.45, '$x_t^{D}$')
# estimation -> decision
arrow([(B2[1], 64.0), (B3[0], 64.0)])
xm2 = (B2[1] + B3[0]) / 2
label(xm2, 64.5, '$\\Delta_t$')
label(xm2, 67.1, '$\\Gamma_t$')
label(xm2, 69.7, '$\\hat{\\sigma}_t$')
# decision -> execution: control action
arrow([(B3[1], 61.0), (B4[0], 61.0)])
label((B3[1] + B4[0]) / 2, 61.5, '$u_t$')
# one-step memory of the hedge position: phi_t -> phi_{t-1}
ax.plot([80.6, 80.6, 66.0], [YT, 77.8, 77.8], color=K, lw=LW, zorder=4)
arrow([(66.0, 77.8), (66.0, YT)])
label(81.2, 75.0, '$\\varphi_t$', ha='left', va='bottom')
label(65.4, 75.0, '$\\varphi_{t-1}$', ha='right', va='bottom')
label(73.3, 78.1, 'задержка на 1 ч', ha='center', va='bottom', fs=5.8, c=G)

# ------------------------------------------------------------------ offline evaluation (test sample)
P0, P1 = 1.0, 34.6
ax.add_patch(Rectangle((0.4, P0), WMM - 0.8, P1 - P0, lw=LW, ec=K, fc=PANEL, ls=(0, (3, 2)), zorder=1))
ax.text(1.6, P1 - 1.4, 'Офлайн-оценка', ha='left', va='top', fontsize=FS_T, fontweight='bold')
for k, s_ in enumerate(['на тестовой выборке:', 'выполняется после', 'наблюдения часа $t+1$', 'и не влияет на решения']):
    ax.text(1.6, P1 - 4.8 - 2.8 * k, s_, ha='left', va='top', fontsize=FS, color=G)

RA0, RA1 = 18.4, 31.6                # row A
RB0, RB1 = 2.4, 15.2                 # row B
E1 = (28.8, 55.6)
E2 = (58.0, 107.0)
box(E1[0], E1[1], RA0, RA1, 'Точность оценивания', [
    'RMSE оценивания IV;',
    'цели: маркировочная IV',
    'и IV котировки (табл. 1)',
], top_pad=1.2, step=2.85, title_gap=3.4)
box(E2[0], E2[1], RA0, RA1, 'Результат шага (4)', [
    ('$\\ell_{t+1}=(C_{t+1}-C_t)-\\varphi_t(S_{t+1}-S_t)-c_t|u_t|S_t$', {'step': 3.3}),
    ('$C_{t+1}$, $S_{t+1}$ – реализованные цены опциона', {'c': G, 'step': 2.8}),
    ('и спота в конце часа $t+1$', {'c': G}),
], top_pad=1.2, title_gap=3.4)
box(1.6, 55.6, RB0, RB1, 'Ценность информации (6)', [
    ('$\\Delta J^{E}=J(f(x^{G}),\\pi)-J(f(x),\\pi)$, при том же $\\pi$', {'step': 3.4}),
    '$\\Delta J^{D}=J(f,\\pi(\\cdot,\\varnothing))-J(f,\\pi(\\cdot,x^{D}))$, при той же $f$',
], top_pad=1.2, title_gap=3.6)
box(E2[0], E2[1], RB0, RB1, 'Показатели (5)', [
    ('$J=(\\mathrm{RMSE}(\\ell),\\ \\mathrm{VaR}_{95}(\\ell),\\ \\mathrm{ES}_{95}(\\ell),\\ \\Sigma_t\\, c_t|u_t|S_t)$', {'step': 3.4}),
    ('по всем шагам теста; фронты Парето', {'c': G}),
], top_pad=1.2, title_gap=3.6)

# online -> offline (dashed): sigma-hat to estimation accuracy, (phi_t, u_t) to the step result
xs = 46.0
arrow([(xs, YB), (xs, RA1)], ls=(0, (2.2, 1.6)))
label(xs + 0.8, 39.0, '$\\hat{\\sigma}_t$', ha='left', va='center', bg=True)
xu = 78.0
arrow([(xu, YB), (xu, RA1)], ls=(0, (2.2, 1.6)))
label(xu + 0.8, 39.0, '$\\varphi_t,\\ u_t$', ha='left', va='center', bg=True)
# offline flow: step result -> J -> value of information
arrow([(82.5, RA0), (82.5, RB1)])
arrow([(E2[0], (RB0 + RB1) / 2), (55.6, (RB0 + RB1) / 2)])

for ext in ('png', 'pdf', 'svg'):
    fig.savefig(os.path.join(OUT, 'Fig1.' + ext), dpi=600, facecolor='white')
plt.close(fig)
print('saved Fig1.{png,pdf,svg} to', OUT)
