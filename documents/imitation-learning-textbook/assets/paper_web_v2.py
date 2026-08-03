import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np

fig, ax = plt.subplots(figsize=(26, 18), facecolor='#0D1117')
ax.set_facecolor('#0D1117')
ax.axis('off')
ax.set_xlim(-1, 26)
ax.set_ylim(-1, 18)

# ──────────────────────────────────────────────
# 1. Define nodes with explicit (x, y) positions
# ──────────────────────────────────────────────
nodes = {
    # Core
    "ACT\n(Zhao 2023)":          {"pos": (13,  9),  "c": "#E84040", "r": 0.95},

    # Parents  – left side, spread vertically
    "BERT\n(Devlin 2019)":       {"pos": ( 3, 16),  "c": "#4488FF", "r": 0.72},
    "VAE/CVAE\n(Kingma 2013)":   {"pos": ( 3, 14),  "c": "#4488FF", "r": 0.72},
    "Attention\n(Vaswani 2017)": {"pos": ( 3, 12),  "c": "#4488FF", "r": 0.72},
    "ResNet\n(He 2016)":         {"pos": ( 3, 10),  "c": "#4488FF", "r": 0.72},
    "DAgger\n(Ross 2011)":       {"pos": ( 3,  8),  "c": "#4488FF", "r": 0.72},
    "ALVINN\n(Pomerleau 1988)":  {"pos": ( 3,  6),  "c": "#4488FF", "r": 0.72},
    "RT-1\n(Brohan 2022)":       {"pos": ( 3,  4),  "c": "#4488FF", "r": 0.72},
    "IBC\n(Florence 2021)":      {"pos": ( 7, 16),  "c": "#4488FF", "r": 0.72},
    "BeT\n(Shafiullah 2022)":    {"pos": ( 7, 14),  "c": "#4488FF", "r": 0.72},
    "MuJoCo\n(Todorov 2012)":    {"pos": ( 7, 12),  "c": "#4488FF", "r": 0.72},
    "BC-Z\n(Jang 2022)":         {"pos": ( 7, 10),  "c": "#4488FF", "r": 0.72},
    "DETR\n(Carion 2020)":       {"pos": ( 7,  8),  "c": "#4488FF", "r": 0.72},
    "Action Chunk\n(Lai 2022)":  {"pos": ( 7,  6),  "c": "#4488FF", "r": 0.72},
    "Bridge Data\n(Ebert 2021)": {"pos": ( 7,  4),  "c": "#4488FF", "r": 0.72},
    "Causal Conf.\n(de Haan 2019)":{"pos":( 7, 2),  "c": "#4488FF", "r": 0.72},

    # Children – right side
    "ACT2/ALOHA2\n(Zhao 2024)":  {"pos": (20, 16),  "c": "#44BB55", "r": 0.72},
    "Diffusion\nPolicy (Chi 2023)":{"pos":(20, 14),  "c": "#44BB55", "r": 0.72},
    "Mobile ALOHA\n(Fu 2024)":   {"pos": (20, 12),  "c": "#44BB55", "r": 0.72},
    "π₀\n(Black 2024)":          {"pos": (20, 10),  "c": "#44BB55", "r": 0.72},
    "CogACT\n(Li 2024)":         {"pos": (20,  8),  "c": "#44BB55", "r": 0.72},
    "InterACT\n(Lee 2024)":      {"pos": (20,  6),  "c": "#44BB55", "r": 0.72},
    "Bi-ACT\n(Buamanee 2024)":   {"pos": (20,  4),  "c": "#44BB55", "r": 0.72},
    "RDT-1B\n(Liu 2024)":        {"pos": (20,  2),  "c": "#44BB55", "r": 0.72},

    # VLA ecosystem – far right
    "RT-2\n(Brohan 2023)":       {"pos": (24, 15),  "c": "#FF8800", "r": 0.72},
    "OpenVLA\n(Kim 2024)":       {"pos": (24, 12),  "c": "#FF8800", "r": 0.72},
    "Octo\n(Ghosh 2024)":        {"pos": (24,  9),  "c": "#FF8800", "r": 0.72},
    "π₀.5\n(PI 2025)":           {"pos": (24,  6),  "c": "#FF8800", "r": 0.72},
    "GATO\n(Reed 2022)":         {"pos": (24,  3),  "c": "#FF8800", "r": 0.72},

    # Data / hardware – bottom center
    "OXE\n(2023)":               {"pos": (13,  1.5), "c": "#AA44FF", "r": 0.72},
    "ALOHA HW\n(2023)":          {"pos": (10,  1.5), "c": "#AA44FF", "r": 0.72},

    # Foundations – top center
    "Behav. Cloning":            {"pos": (13, 16.5), "c": "#888888", "r": 0.68},
    "GAIL\n(Ho 2016)":           {"pos": (10, 16.5), "c": "#888888", "r": 0.68},
    "Decision\nTransformer":     {"pos": (16, 16.5), "c": "#888888", "r": 0.68},
}

# ──────────────────────────────────────────────
# 2. Draw edges  (draw first, under nodes)
# ──────────────────────────────────────────────
act_pos = nodes["ACT\n(Zhao 2023)"]["pos"]

def draw_edge(ax, p1, p2, color, lw=1.0, alpha=0.55, rad=0.0):
    ax.annotate("", xy=p2, xytext=p1,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, alpha=alpha,
                                connectionstyle=f"arc3,rad={rad}"),
                annotation_clip=False, zorder=1)

# ACT → parents
parent_names = [
    "BERT\n(Devlin 2019)", "VAE/CVAE\n(Kingma 2013)", "Attention\n(Vaswani 2017)",
    "ResNet\n(He 2016)", "DAgger\n(Ross 2011)", "ALVINN\n(Pomerleau 1988)",
    "RT-1\n(Brohan 2022)", "IBC\n(Florence 2021)", "BeT\n(Shafiullah 2022)",
    "MuJoCo\n(Todorov 2012)", "BC-Z\n(Jang 2022)", "DETR\n(Carion 2020)",
    "Action Chunk\n(Lai 2022)", "Bridge Data\n(Ebert 2021)",
    "Causal Conf.\n(de Haan 2019)", "ALOHA HW\n(2023)"
]
for n in parent_names:
    draw_edge(ax, act_pos, nodes[n]["pos"], "#5599FF", lw=1.1, alpha=0.5)

# children → ACT
child_names = [
    "ACT2/ALOHA2\n(Zhao 2024)", "Diffusion\nPolicy (Chi 2023)", "Mobile ALOHA\n(Fu 2024)",
    "π₀\n(Black 2024)", "CogACT\n(Li 2024)", "InterACT\n(Lee 2024)",
    "Bi-ACT\n(Buamanee 2024)", "RDT-1B\n(Liu 2024)"
]
for n in child_names:
    draw_edge(ax, nodes[n]["pos"], act_pos, "#44CC66", lw=1.1, alpha=0.5)

# VLA cross-links
vla_links = [
    ("RT-2\n(Brohan 2023)",  "RT-1\n(Brohan 2022)"),
    ("OpenVLA\n(Kim 2024)",  "RT-2\n(Brohan 2023)"),
    ("OpenVLA\n(Kim 2024)",  "OXE\n(2023)"),
    ("Octo\n(Ghosh 2024)",   "OXE\n(2023)"),
    ("π₀\n(Black 2024)",     "OpenVLA\n(Kim 2024)"),
    ("π₀\n(Black 2024)",     "Diffusion\nPolicy (Chi 2023)"),
    ("π₀.5\n(PI 2025)",      "π₀\n(Black 2024)"),
    ("CogACT\n(Li 2024)",    "OpenVLA\n(Kim 2024)"),
    ("OXE\n(2023)",          "RT-1\n(Brohan 2022)"),
    ("RT-1\n(Brohan 2022)",  "Behav. Cloning"),
]
for (a, b) in vla_links:
    draw_edge(ax, nodes[a]["pos"], nodes[b]["pos"], "#FFAA44", lw=0.9, alpha=0.4, rad=0.1)

# ──────────────────────────────────────────────
# 3. Draw nodes
# ──────────────────────────────────────────────
for name, info in nodes.items():
    x, y = info["pos"]
    color = info["c"]
    r = info["r"]
    # Glow
    circle_glow = plt.Circle((x, y), r * 1.5, color=color, alpha=0.10, zorder=2)
    ax.add_patch(circle_glow)
    # Body
    circle = plt.Circle((x, y), r, color=color, alpha=0.90, zorder=3,
                         linewidth=0.7, edgecolor='white')
    ax.add_patch(circle)
    # Label
    fontsize = 8.5 if name == "ACT\n(Zhao 2023)" else 6.5
    fontweight = 'bold' if name == "ACT\n(Zhao 2023)" else 'normal'
    ax.text(x, y, name, ha='center', va='center',
            fontsize=fontsize, color='white', fontweight=fontweight,
            zorder=4, linespacing=1.3)

# ──────────────────────────────────────────────
# 4. Section labels (non-overlapping banners)
# ──────────────────────────────────────────────
banner_style = dict(boxstyle='round,pad=0.35', facecolor='#1A1A2E',
                    edgecolor='#444466', alpha=0.85)

ax.text(5.0, 17.4, "Papers ACT Cites", fontsize=9, color='#88AAFF',
        ha='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#0D1131', edgecolor='#4488FF', alpha=0.8))
ax.text(20.0, 17.4, "Papers Citing ACT", fontsize=9, color='#88EE88',
        ha='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#0D2011', edgecolor='#44BB55', alpha=0.8))
ax.text(24.0, 17.4, "VLA Ecosystem", fontsize=9, color='#FFBB55',
        ha='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#201100', edgecolor='#FF8800', alpha=0.8))
ax.text(11.5, 17.4, "Foundations", fontsize=9, color='#AAAAAA',
        ha='center', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#111111', edgecolor='#888888', alpha=0.8))

# ──────────────────────────────────────────────
# 5. Legend
# ──────────────────────────────────────────────
legend_items = [
    mpatches.Patch(color="#E84040", label="ACT — Core Paper"),
    mpatches.Patch(color="#4488FF", label="Papers ACT Cites"),
    mpatches.Patch(color="#44BB55", label="Papers Citing ACT"),
    mpatches.Patch(color="#FF8800", label="VLA Ecosystem"),
    mpatches.Patch(color="#AA44FF", label="Data & Hardware"),
    mpatches.Patch(color="#888888", label="Foundations"),
]
leg = ax.legend(handles=legend_items, loc='lower left', framealpha=0.5,
                facecolor='#1A1A2E', edgecolor='#555577',
                labelcolor='white', fontsize=8.5,
                title='Node Clusters', title_fontsize=9)
leg.get_title().set_color('#CCCCEE')

# Title
ax.set_title("Imitation Learning Paper Web  ·  ACT & VLA Ecosystem",
             fontsize=14, color='white', fontweight='bold', pad=12)

plt.tight_layout()
plt.savefig('/home/user/workspace/imitation-learning-textbook/assets/paper_web.png',
            dpi=150, bbox_inches='tight', facecolor='#0D1117')
print("Done.")
