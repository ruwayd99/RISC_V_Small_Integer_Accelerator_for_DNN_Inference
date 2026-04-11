import matplotlib.pyplot as plt
import numpy as np

data = {
    "mul+add": {"runtime": 382, "alm": 566,  "dsp": 3},
    "mac":     {"runtime": 318, "alm": 597,  "dsp": 3},
    "dot4":    {"runtime": 194, "alm": 581,  "dsp": 6},
    "mmatmul": {"runtime": 44,  "alm": 1040, "dsp": 19}
}

names = list(data.keys())
runtimes = np.array([d["runtime"] for d in data.values()])
alms = np.array([d["alm"] for d in data.values()])
dsps = np.array([d["dsp"] for d in data.values()])

def get_diagonal_pareto(xs, ys):
    pts = sorted(zip(xs, ys))
    if not pts: return [], []
    pareto_pts = []
    curr_min_y = float('inf')
    for x, y in pts:
        if y < curr_min_y:
            pareto_pts.append((x, y))
            curr_min_y = y
    if len(pareto_pts) < 3: return zip(*pareto_pts)
    final_pts = [pareto_pts[0]]
    for i in range(1, len(pareto_pts) - 1):
        x1, y1 = final_pts[-1]
        x2, y2 = pareto_pts[i]
        x3, y3 = pareto_pts[i+1]
        line_y = y1 + (y3 - y1) * (x2 - x1) / (x3 - x1)
        if y2 <= line_y:
            final_pts.append((x2, y2))
    final_pts.append(pareto_pts[-1])
    return zip(*final_pts)

alm_px, alm_py = get_diagonal_pareto(alms, runtimes)
dsp_px, dsp_py = get_diagonal_pareto(dsps, runtimes)

fig, ax1 = plt.subplots(figsize=(10, 12))

FONT_SIZE_LABEL = 38  # was 32
FONT_SIZE_TICK = 30   # was 26
FONT_SIZE_ANNOT = 27  # was 22

bbox_alm = dict(boxstyle="round,pad=0.2", fc="white", ec="tab:red", lw=2)
bbox_dsp = dict(boxstyle="round,pad=0.2", fc="white", ec="tab:blue", lw=2)

ax1.set_xlabel('AREA (ALMs)', color='tab:red', fontsize=FONT_SIZE_LABEL, fontweight='bold', labelpad=20)
ax1.set_ylabel('Runtime (cycles)', fontsize=FONT_SIZE_LABEL, fontweight='bold')
ax1.tick_params(axis='both', labelsize=FONT_SIZE_TICK)
ax1.tick_params(axis='x', labelcolor='tab:red')

alm_scat = ax1.scatter(alms, runtimes, facecolors='none', edgecolors='tab:red',
                       marker='o', s=500, lw=3, label='ALM Designs', zorder=5)
ax1.plot(list(alm_px), list(alm_py), color='tab:red', linestyle='--', lw=4, alpha=0.7, zorder=4)

ax2 = ax1.twiny()
ax2.set_xlabel('AREA (DSPs)', color='tab:blue', fontsize=FONT_SIZE_LABEL, fontweight='bold', labelpad=20)
ax2.tick_params(axis='x', labelsize=FONT_SIZE_TICK, labelcolor='tab:blue')

dsp_scat = ax2.scatter(dsps, runtimes, facecolors='none', edgecolors='tab:blue',
                       marker='s', s=500, lw=3, label='DSP Designs', zorder=5)
ax2.plot(list(dsp_px), list(dsp_py), color='tab:blue', linestyle='--', lw=4, alpha=0.7, zorder=4)

# Per-point label offsets: (alm_dx, alm_dy, alm_ha, dsp_dx, dsp_dy, dsp_ha)
label_cfg = {
    "mul+add": ( 18,  -10, 'left',   80,  40, 'right'),
    "mac":     ( 18,    6, 'left',   18, -40, 'right'),
    "dot4":    ( 18,  -40, 'right',  18,  20, 'left' ),
    "mmatmul": (-18,   25, 'left',  -18, -40, 'left'),
}

for i, txt in enumerate(names):
    adx, ady, aha, ddx, ddy, dha = label_cfg[txt]
    ax1.annotate(txt.lower(), (alms[i], runtimes[i]),
                 xytext=(adx, ady), textcoords='offset points',
                 color='tab:red', fontweight='bold', fontsize=FONT_SIZE_ANNOT,
                 bbox=bbox_alm, zorder=12, ha=aha)
    ax2.annotate(txt.upper(), (dsps[i], runtimes[i]),
                 xytext=(ddx, ddy), textcoords='offset points',
                 color='tab:blue', fontweight='bold', fontsize=FONT_SIZE_ANNOT,
                 bbox=bbox_dsp, zorder=11, ha=dha)

handles = [alm_scat, dsp_scat]
labels = [h.get_label() for h in handles]
ax1.legend(handles, labels, loc='upper right', fontsize=26, frameon=True, shadow=True)  # was 22

ax1.set_xlim(500, 1150)
ax2.set_xlim(2, 22)
ax1.set_ylim(-50, 480)
ax1.grid(True, linestyle=':', alpha=0.5)

plt.subplots_adjust(top=0.82, bottom=0.15, left=0.18, right=0.92)
plt.show()