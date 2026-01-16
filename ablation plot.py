
import matplotlib.pyplot as plt
import numpy as np

# Updated method names
methods = [
    "Smile",
    "Smile w/o Empty Classifier",
    "Smile w/o Empty Classifier & Similairty Thresold",
    "Smile w/ gk JB stats only"
]

# Updated metric names
metrics = ['Gm', 'Mcc', r'$F_1$', 'Acc']

# Metric data: shape is [method][metric]
metric_avg = np.array([
    [0.8442, 0.6922, 0.8422, 0.8450],  # Smile
    [0.8379, 0.6776, 0.8380, 0.8383],  # Smile w/o Empty Classifier
    [0.6751, 0.4677, 0.7623, 0.7100],  # Smile w/o Empty Classifier & Similairty Thresold
    [0.7870, 0.5854, 0.8020, 0.7900]   # Smile w/ JB stats only
])

metric_std = np.array([
    [0.0217, 0.0482, 0.0184, 0.0227],
    [0.0243, 0.0500, 0.0208, 0.0246],
    [0.0000, 0.0000, 0.0000, 0.0000],
    [0.0138, 0.0222, 0.0080, 0.0122]
])

# Plot settings
bar_width = 0.18
x = np.arange(len(metrics))  # positions for metric groups

colors = ['gray', 'lightgray', 'silver', 'darkgray']
hatches = ['/', '\\', 'xx', '///']

fig, ax = plt.subplots(figsize=(7, 3.5))  # suitable for IEEE column width

# Plot each method's bar across all metrics
for i in range(len(methods)):
    ax.bar(
        x + i * bar_width,
        metric_avg[i],
        yerr=metric_std[i],
        width=bar_width,
        label=methods[i],
        color=colors[i % len(colors)],
        edgecolor='black',
        hatch=hatches[i % len(hatches)],
        capsize=3,
        linewidth=0.8
    )

# Axis & legend formatting
ax.set_ylabel('Score', fontsize=10)
ax.set_xticks(x + 1.5 * bar_width)
ax.set_xticklabels(metrics, fontsize=9)
ax.set_ylim(0.4, 1.0)
ax.legend(
    fontsize=8,
    ncol=2,
    loc='upper center',
    bbox_to_anchor=(0.5, 1.25),
    frameon=False
)

# Aesthetic cleanup
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
plt.tight_layout()

# Save in vector and raster format
plt.savefig("ablation.pdf", format='pdf', dpi=300, bbox_inches='tight')
plt.savefig("ablation.png", format='png', dpi=300, bbox_inches='tight')
plt.show()