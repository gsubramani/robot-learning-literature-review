import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Paper web data structure
papers = {
    # Core ACT paper
    "ACT\n(Zhao et al. 2023)": {"cluster": "core", "year": 2023},
    
    # Papers ACT cites (parents)
    "BERT\n(Devlin 2019)": {"cluster": "act_parents", "year": 2019},
    "VAE/CVAE\n(Kingma 2013)": {"cluster": "act_parents", "year": 2013},
    "Attention Is\nAll You Need\n(Vaswani 2017)": {"cluster": "act_parents", "year": 2017},
    "ResNet\n(He 2016)": {"cluster": "act_parents", "year": 2016},
    "DAgger\n(Ross 2011)": {"cluster": "act_parents", "year": 2011},
    "RT-1\n(Brohan 2022)": {"cluster": "act_parents", "year": 2022},
    "IBC\n(Florence 2021)": {"cluster": "act_parents", "year": 2021},
    "BeT\n(Shafiullah 2022)": {"cluster": "act_parents", "year": 2022},
    "MuJoCo\n(Todorov 2012)": {"cluster": "act_parents", "year": 2012},
    "BC-Z\n(Jang 2022)": {"cluster": "act_parents", "year": 2022},
    "Action Chunking\n(Lai 2022)": {"cluster": "act_parents", "year": 2022},
    "ALVINN\n(Pomerleau 1988)": {"cluster": "act_parents", "year": 1988},
    "Bridge Data\n(Ebert 2021)": {"cluster": "act_parents", "year": 2021},
    "Causal Confusion\n(de Haan 2019)": {"cluster": "act_parents", "year": 2019},
    "DETR\n(Carion 2020)": {"cluster": "act_parents", "year": 2020},
    
    # Papers citing ACT (children)
    "ACT2 / ALOHA2\n(Zhao 2024)": {"cluster": "act_children", "year": 2024},
    "Diffusion Policy\n(Chi 2023)": {"cluster": "act_children", "year": 2023},
    "Mobile ALOHA\n(Fu 2024)": {"cluster": "act_children", "year": 2024},
    "π₀\n(Black 2024)": {"cluster": "act_children", "year": 2024},
    "CogACT\n(Li 2024)": {"cluster": "act_children", "year": 2024},
    "InterACT\n(Lee 2024)": {"cluster": "act_children", "year": 2024},
    "Bi-ACT\n(Buamanee 2024)": {"cluster": "act_children", "year": 2024},
    "RDT-1B\n(Liu 2024)": {"cluster": "act_children", "year": 2024},
    "Bidirectional\nDecoding (Liu 2024)": {"cluster": "act_children", "year": 2024},
    
    # VLA Ecosystem
    "RT-2\n(Brohan 2023)": {"cluster": "vla", "year": 2023},
    "OpenVLA\n(Kim 2024)": {"cluster": "vla", "year": 2024},
    "Octo\n(Ghosh 2024)": {"cluster": "vla", "year": 2024},
    "CoT-VLA\n(Zhao 2025)": {"cluster": "vla", "year": 2025},
    "π₀.5\n(PI 2025)": {"cluster": "vla", "year": 2025},
    "GATO\n(Reed 2022)": {"cluster": "vla", "year": 2022},
    
    # Foundation data
    "Open X-Emb.\n(OXE 2023)": {"cluster": "data", "year": 2023},
    "ALOHA Hardware\n(Zhao 2023)": {"cluster": "data", "year": 2023},
    
    # Related methods
    "Behavior\nCloning": {"cluster": "foundations", "year": 1988},
    "GAIL\n(Ho 2016)": {"cluster": "foundations", "year": 2016},
    "Decision\nTransformer\n(Chen 2021)": {"cluster": "foundations", "year": 2021},
}

# Define edges (source -> target means "cites" or "builds on")
edges = [
    # ACT cites these
    ("ACT\n(Zhao et al. 2023)", "BERT\n(Devlin 2019)"),
    ("ACT\n(Zhao et al. 2023)", "VAE/CVAE\n(Kingma 2013)"),
    ("ACT\n(Zhao et al. 2023)", "Attention Is\nAll You Need\n(Vaswani 2017)"),
    ("ACT\n(Zhao et al. 2023)", "ResNet\n(He 2016)"),
    ("ACT\n(Zhao et al. 2023)", "DAgger\n(Ross 2011)"),
    ("ACT\n(Zhao et al. 2023)", "RT-1\n(Brohan 2022)"),
    ("ACT\n(Zhao et al. 2023)", "IBC\n(Florence 2021)"),
    ("ACT\n(Zhao et al. 2023)", "BeT\n(Shafiullah 2022)"),
    ("ACT\n(Zhao et al. 2023)", "MuJoCo\n(Todorov 2012)"),
    ("ACT\n(Zhao et al. 2023)", "BC-Z\n(Jang 2022)"),
    ("ACT\n(Zhao et al. 2023)", "Action Chunking\n(Lai 2022)"),
    ("ACT\n(Zhao et al. 2023)", "ALVINN\n(Pomerleau 1988)"),
    ("ACT\n(Zhao et al. 2023)", "Bridge Data\n(Ebert 2021)"),
    ("ACT\n(Zhao et al. 2023)", "Causal Confusion\n(de Haan 2019)"),
    ("ACT\n(Zhao et al. 2023)", "DETR\n(Carion 2020)"),
    ("ACT\n(Zhao et al. 2023)", "ALOHA Hardware\n(Zhao 2023)"),
    # Papers citing ACT
    ("ACT2 / ALOHA2\n(Zhao 2024)", "ACT\n(Zhao et al. 2023)"),
    ("Diffusion Policy\n(Chi 2023)", "ACT\n(Zhao et al. 2023)"),
    ("Mobile ALOHA\n(Fu 2024)", "ACT\n(Zhao et al. 2023)"),
    ("π₀\n(Black 2024)", "ACT\n(Zhao et al. 2023)"),
    ("CogACT\n(Li 2024)", "ACT\n(Zhao et al. 2023)"),
    ("InterACT\n(Lee 2024)", "ACT\n(Zhao et al. 2023)"),
    ("Bi-ACT\n(Buamanee 2024)", "ACT\n(Zhao et al. 2023)"),
    ("RDT-1B\n(Liu 2024)", "ACT\n(Zhao et al. 2023)"),
    ("Bidirectional\nDecoding (Liu 2024)", "ACT\n(Zhao et al. 2023)"),
    # VLA connections
    ("RT-2\n(Brohan 2023)", "RT-1\n(Brohan 2022)"),
    ("OpenVLA\n(Kim 2024)", "RT-2\n(Brohan 2023)"),
    ("OpenVLA\n(Kim 2024)", "Open X-Emb.\n(OXE 2023)"),
    ("OpenVLA\n(Kim 2024)", "Octo\n(Ghosh 2024)"),
    ("Octo\n(Ghosh 2024)", "Open X-Emb.\n(OXE 2023)"),
    ("π₀\n(Black 2024)", "OpenVLA\n(Kim 2024)"),
    ("π₀\n(Black 2024)", "Diffusion Policy\n(Chi 2023)"),
    ("π₀.5\n(PI 2025)", "π₀\n(Black 2024)"),
    ("CoT-VLA\n(Zhao 2025)", "OpenVLA\n(Kim 2024)"),
    ("GATO\n(Reed 2022)", "Decision\nTransformer\n(Chen 2021)"),
    ("RT-1\n(Brohan 2022)", "Behavior\nCloning"),
    ("Diffusion Policy\n(Chi 2023)", "Behavior\nCloning"),
    ("Open X-Emb.\n(OXE 2023)", "RT-1\n(Brohan 2022)"),
    ("CogACT\n(Li 2024)", "OpenVLA\n(Kim 2024)"),
]

# Cluster positions (arranged in radial layout)
cluster_centers = {
    "core": (0, 0),
    "act_parents": None,  # will be computed radially
    "act_children": None,
    "vla": (3.5, 1.5),
    "data": (3.5, -1.5),
    "foundations": (-4, -2),
}

cluster_colors = {
    "core": "#FF4444",
    "act_parents": "#4488FF",
    "act_children": "#44BB44",
    "vla": "#FF8800",
    "data": "#AA44FF",
    "foundations": "#888888",
}

cluster_labels = {
    "core": "ACT (Core Paper)",
    "act_parents": "Papers ACT Cites",
    "act_children": "Papers Citing ACT",
    "vla": "VLA Ecosystem",
    "data": "Data & Hardware",
    "foundations": "Foundations",
}

fig, ax = plt.subplots(1, 1, figsize=(22, 16), facecolor='#0D1117')
ax.set_facecolor('#0D1117')
ax.axis('off')

# Compute positions
positions = {}

# Core
positions["ACT\n(Zhao et al. 2023)"] = np.array([0, 0])

# Act parents - inner ring left/top-left
parent_papers = [k for k, v in papers.items() if v["cluster"] == "act_parents"]
angles_parents = np.linspace(100, 260, len(parent_papers))
for i, p in enumerate(parent_papers):
    r = 3.8
    ang = np.radians(angles_parents[i])
    positions[p] = np.array([r * np.cos(ang), r * np.sin(ang)])

# Act children - right side
child_papers = [k for k, v in papers.items() if v["cluster"] == "act_children"]
angles_children = np.linspace(-60, 60, len(child_papers))
for i, p in enumerate(child_papers):
    r = 3.5
    ang = np.radians(angles_children[i])
    positions[p] = np.array([r * np.cos(ang), r * np.sin(ang)])

# VLA ecosystem
vla_papers = [k for k, v in papers.items() if v["cluster"] == "vla"]
vla_base = np.array([5.5, 0])
vla_offsets = [
    (-0.5, 2.2), (0.5, 1.0), (0.5, -1.0), (-0.5, -2.2), (1.5, 0), (-1.5, 0)
]
for i, p in enumerate(vla_papers[:len(vla_offsets)]):
    positions[p] = vla_base + np.array(vla_offsets[i])

# Data
data_papers = [k for k, v in papers.items() if v["cluster"] == "data"]
for i, p in enumerate(data_papers):
    positions[p] = np.array([2.5, -3.5 + i * 1.2])

# Foundations
found_papers = [k for k, v in papers.items() if v["cluster"] == "foundations"]
for i, p in enumerate(found_papers):
    positions[p] = np.array([-5.5, 1.5 - i * 1.5])

# Draw edges
for (src, tgt) in edges:
    if src in positions and tgt in positions:
        p1 = positions[src]
        p2 = positions[tgt]
        # Color by direction
        src_cluster = papers[src]["cluster"]
        tgt_cluster = papers[tgt]["cluster"]
        if tgt_cluster == "core" or src_cluster == "core":
            color = "#FF666688"
            lw = 1.2
        else:
            color = "#FFFFFF22"
            lw = 0.7
        ax.annotate("", xy=p2, xytext=p1,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                   connectionstyle="arc3,rad=0.05"),
                    annotation_clip=False)

# Draw nodes
node_sizes = {"core": 2200, "act_parents": 800, "act_children": 950,
              "vla": 900, "data": 850, "foundations": 800}
font_sizes = {"core": 9, "act_parents": 6.2, "act_children": 6.5,
              "vla": 6.5, "data": 6.5, "foundations": 6.5}

for name, pos in positions.items():
    cluster = papers[name]["cluster"]
    color = cluster_colors[cluster]
    size = node_sizes[cluster]
    fsize = font_sizes[cluster]
    
    # Draw glow effect
    ax.scatter(*pos, s=size * 2.5, c=color, alpha=0.12, zorder=2)
    ax.scatter(*pos, s=size, c=color, alpha=0.85, zorder=3,
               edgecolors='white', linewidths=0.5)
    
    # Label
    ax.text(pos[0], pos[1], name, ha='center', va='center',
            fontsize=fsize, color='white', fontweight='bold',
            zorder=4, wrap=True,
            bbox=dict(boxstyle='round,pad=0.1', facecolor='none', edgecolor='none'))

# Legend
legend_patches = [mpatches.Patch(color=v, label=cluster_labels[k])
                  for k, v in cluster_colors.items()]
legend = ax.legend(handles=legend_patches, loc='lower left',
                   framealpha=0.3, facecolor='#1A1A2E',
                   edgecolor='#444444', labelcolor='white',
                   fontsize=9, title='Paper Clusters',
                   title_fontsize=10)
legend.get_title().set_color('white')

ax.set_title('Imitation Learning Paper Web\nACT (Action Chunking with Transformers) & VLA Ecosystem',
             fontsize=15, color='white', fontweight='bold', pad=20)

# Year annotation for core
core_pos = positions["ACT\n(Zhao et al. 2023)"]
ax.text(core_pos[0], core_pos[1] - 0.42, "RSS 2023",
        ha='center', fontsize=7, color='#FFDDAA', zorder=5)

ax.set_xlim(-7.5, 8.5)
ax.set_ylim(-5.5, 5.5)

plt.tight_layout()
plt.savefig('/home/user/workspace/imitation-learning-textbook/assets/paper_web.png',
            dpi=160, bbox_inches='tight', facecolor='#0D1117')
print("Paper web saved.")
