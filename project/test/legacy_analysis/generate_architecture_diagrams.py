"""
Visual Architecture Diagram Generator
Creates a flowchart showing data flow with actual dimensions
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(16, 24))
ax.set_xlim(0, 10)
ax.set_ylim(0, 30)
ax.axis('off')

# Colors
color_data = '#E8F4F8'
color_backbone = '#FFE8D6'
color_trainable = '#D4EDDA'
color_anchor = '#F8D7DA'
color_loss = '#D6D8F5'
color_inference = '#FFF3CD'

def draw_box(ax, x, y, w, h, text, color, fontsize=10, bold=False):
    """Draw a box with text"""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        edgecolor='black',
        facecolor=color,
        linewidth=2
    )
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x + w/2, y + h/2, text, 
            ha='center', va='center', fontsize=fontsize, weight=weight,
            wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, label='', style='->', width=2):
    """Draw an arrow between boxes"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color='black',
        linewidth=width,
        mutation_scale=20
    )
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Title
ax.text(5, 29, 'BMAD Architecture: Complete Data Flow', 
        ha='center', fontsize=18, weight='bold')
ax.text(5, 28.3, 'Brain MRI Anomaly Detection with DINOv3', 
        ha='center', fontsize=12, style='italic')

y_pos = 27

# ============================================================================
# PART 1: INPUT DATA
# ============================================================================
ax.text(5, y_pos, '1. INPUT DATA', ha='center', fontsize=14, weight='bold')
y_pos -= 0.8

draw_box(ax, 0.5, y_pos-1.5, 4, 1.3, 
         'Raw FLAIR MRI\n240×240 grayscale\nBraTS2021 dataset', 
         color_data, fontsize=10, bold=True)

draw_box(ax, 5.5, y_pos-1.5, 4, 1.3,
         'Preprocessing\n• Percentile clipping\n• Z-score normalize\n• Augmentation',
         color_data, fontsize=9)

draw_arrow(ax, 4.5, y_pos-0.85, 5.5, y_pos-0.85)

y_pos -= 2

draw_box(ax, 2, y_pos-1.5, 6, 1.3,
         'Batch Tensor\n(64, 3, 240, 240)\nbatch × RGB × height × width',
         color_data, fontsize=11, bold=True)

draw_arrow(ax, 5, y_pos+0.15, 5, y_pos-0.5)

y_pos -= 2.5

# ============================================================================
# PART 2: DINOV3 BACKBONE (FROZEN)
# ============================================================================
ax.text(5, y_pos, '2. FROZEN DINOv3 BACKBONE', ha='center', fontsize=14, weight='bold')
y_pos -= 0.8

draw_box(ax, 1, y_pos-2, 8, 1.8,
         'vit_small_patch16_dinov3.lvd1689m\n' + 
         '22M parameters (FROZEN)\n' +
         'Patch size: 16×16 → Grid: 15×15 = 225 patches\n' +
         'Embedding dimension: 384',
         color_backbone, fontsize=10, bold=True)

y_pos -= 2.5

# Token outputs
draw_box(ax, 0.5, y_pos-1.5, 4, 1.3,
         'CLS Token\n(64, 384)\nGlobal representation',
         color_backbone, fontsize=10, bold=True)

draw_box(ax, 5.5, y_pos-1.5, 4, 1.3,
         'Patch Tokens\n(64, 225, 384)\n225 spatial patches',
         color_backbone, fontsize=10, bold=True)

draw_arrow(ax, 2.5, y_pos+0.2, 2.5, y_pos-0.2)
draw_arrow(ax, 7.5, y_pos+0.2, 7.5, y_pos-0.2)

y_pos -= 2.5

# Reshape
draw_box(ax, 5.5, y_pos-1.2, 4, 1,
         'Reshape to spatial\n(64, 15, 15, 384)',
         color_backbone, fontsize=9)

draw_arrow(ax, 7.5, y_pos+0.3, 7.5, y_pos-0.2)

y_pos -= 2

# ============================================================================
# PART 3: TRAINABLE PROJECTION HEAD
# ============================================================================
ax.text(5, y_pos, '3. TRAINABLE PROJECTION HEAD', ha='center', fontsize=14, weight='bold')
y_pos -= 0.8

draw_box(ax, 1.5, y_pos-1.8, 7, 1.6,
         'Two-layer MLP: 384 → 192 → 128\n' +
         '98,624 trainable parameters\n' +
         'Linear → ReLU → Linear',
         color_trainable, fontsize=10, bold=True)

y_pos -= 2.3

# Projected features
draw_box(ax, 0.5, y_pos-1.5, 4, 1.3,
         'Global Features\n(64, 128)\nL2 normalized',
         color_trainable, fontsize=10, bold=True)

draw_box(ax, 5.5, y_pos-1.5, 4, 1.3,
         'Dense Features\n(64, 15, 15, 128)\nPatch embeddings',
         color_trainable, fontsize=10, bold=True)

draw_arrow(ax, 2.5, y_pos+0.2, 2.5, y_pos-0.2)
draw_arrow(ax, 7.5, y_pos+0.2, 7.5, y_pos-0.2)

y_pos -= 2.5

# ============================================================================
# PART 4: ANCHORS (SIDE BRANCH)
# ============================================================================
ax.text(9.5, 22, 'ANCHORS', ha='center', fontsize=12, weight='bold', 
        rotation=90, bbox=dict(boxstyle='round', facecolor=color_anchor, edgecolor='black', linewidth=2))

anchor_y = 20.5
draw_box(ax, 9, anchor_y-0.8, 1, 0.6,
         '8 Anchor\nImages\n(8,240,240)',
         color_anchor, fontsize=7)

anchor_y -= 1.2
draw_arrow(ax, 9.5, anchor_y+0.5, 9.5, anchor_y+0.1, '', '->', 1)

draw_box(ax, 9, anchor_y-0.8, 1, 0.6,
         'DINOv3\n(frozen)\n(8,384)',
         color_anchor, fontsize=7)

anchor_y -= 1.2
draw_arrow(ax, 9.5, anchor_y+0.5, 9.5, anchor_y+0.1, '', '->', 1)

draw_box(ax, 9, anchor_y-0.8, 1, 0.6,
         'Project\n(trainable)\n(8,128)',
         color_anchor, fontsize=7)

anchor_y -= 1.2
draw_arrow(ax, 9.5, anchor_y+0.5, 9.5, anchor_y+0.1, '', '->', 1)

draw_box(ax, 9, anchor_y-0.8, 1, 0.6,
         'Normalize\n||·||=1',
         color_anchor, fontsize=7)

# ============================================================================
# PART 5: DISTANCE COMPUTATION
# ============================================================================
ax.text(5, y_pos, '4. DISTANCE TO ANCHORS', ha='center', fontsize=14, weight='bold')
y_pos -= 0.8

draw_box(ax, 0.5, y_pos-1.5, 4, 1.3,
         'Global Distances\n(64, 8)\nL2: ||sample - anchor||₂\nCosine: 1 - similarity',
         color_loss, fontsize=9, bold=True)

draw_box(ax, 5.5, y_pos-1.5, 4, 1.3,
         'Dense Distances\n(64, 8, 15, 15)\nPatch-wise distances',
         color_loss, fontsize=9, bold=True)

draw_arrow(ax, 2.5, y_pos+0.2, 2.5, y_pos-0.2)
draw_arrow(ax, 7.5, y_pos+0.2, 7.5, y_pos-0.2)

# Connect anchors to distance computation
draw_arrow(ax, 9, y_pos-0.85, 4.5, y_pos-0.85, 'K=8 anchors', '<-', 1.5)

y_pos -= 2.5

# ============================================================================
# PART 6: LOSS (TRAINING)
# ============================================================================
ax.text(2.5, y_pos, '5. TRAINING LOSS', ha='center', fontsize=14, weight='bold')
y_pos -= 0.8

draw_box(ax, 0.5, y_pos-2.5, 4, 2.2,
         'Anchor-Margin Loss\n\n' +
         'Attractor (α=1.0):\n' +
         'L_A = ½·mean(min_dist²)\n' +
         'Pull samples to nearest anchor\n\n' +
         'Repeller (β=0.0): OFF\n' +
         'Push anchors apart',
         color_loss, fontsize=9, bold=True)

draw_arrow(ax, 2.5, y_pos+0.5, 2.5, y_pos-0.3)

y_pos -= 3

draw_box(ax, 0.5, y_pos-1.5, 4, 1.3,
         'Backpropagation\nUpdate projection head\n98,624 parameters',
         color_loss, fontsize=10, bold=True)

y_pos -= 2

# ============================================================================
# PART 7: INFERENCE
# ============================================================================
ax.text(7.5, y_pos+3.5, '6. INFERENCE', ha='center', fontsize=14, weight='bold')
inf_y = y_pos + 2.7

draw_box(ax, 5.5, inf_y-2.5, 4, 2.2,
         'Anomaly Scoring\n\n' +
         'Image-level:\n' +
         'score = min(global_distances)\n' +
         '(64,) → scalar per image\n\n' +
         'Pixel-level:\n' +
         'map = min(dense_distances, dim=1)\n' +
         '(64, 15, 15) → upsample → (64, 240, 240)',
         color_inference, fontsize=9, bold=True)

draw_arrow(ax, 7.5, inf_y+0.4, 7.5, inf_y-0.3)

inf_y -= 3

draw_box(ax, 5.5, inf_y-1.5, 4, 1.3,
         'Evaluation Metrics\n' +
         'Image AUROC: 82.35%\n' +
         'Pixel AUROC: 87.06%',
         color_inference, fontsize=10, bold=True)

# ============================================================================
# LEGEND
# ============================================================================
legend_y = 1.5
legend_items = [
    (color_data, 'Data Processing'),
    (color_backbone, 'Frozen Backbone'),
    (color_trainable, 'Trainable Projection'),
    (color_anchor, 'Anchor Processing'),
    (color_loss, 'Loss & Training'),
    (color_inference, 'Inference & Evaluation')
]

ax.text(5, legend_y + 0.8, 'Legend', ha='center', fontsize=12, weight='bold')

for i, (color, label) in enumerate(legend_items):
    x = 1 + (i % 3) * 2.7
    y = legend_y - (i // 3) * 0.6
    box = FancyBboxPatch(
        (x, y), 0.4, 0.4,
        boxstyle="round,pad=0.05",
        edgecolor='black',
        facecolor=color,
        linewidth=1
    )
    ax.add_patch(box)
    ax.text(x + 0.6, y + 0.2, label, fontsize=9, va='center')

# ============================================================================
# KEY INSIGHTS BOX
# ============================================================================
insights_y = 0.2
insights_box = FancyBboxPatch(
    (0.3, insights_y), 9.4, 0.9,
    boxstyle="round,pad=0.1",
    edgecolor='red',
    facecolor='#FFFACD',
    linewidth=2
)
ax.add_patch(insights_box)

insights_text = (
    "KEY INSIGHTS: "
    "• DINOv3 frozen (22M params) + Trainable projection (98K params)  "
    "• 8 eigenface anchors in 128D space  "
    "• Distance metric: L2 or Cosine  "
    "• Attractor loss only (β=0)  "
    "• 15×15 patch grid for localization"
)
ax.text(5, insights_y + 0.45, insights_text, ha='center', fontsize=9, weight='bold')

plt.tight_layout()
plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight')
print("✅ Architecture diagram saved: architecture_diagram.png")
plt.close()

# ============================================================================
# SECOND DIAGRAM: EMBEDDING SPACES
# ============================================================================
fig2, axes = plt.subplots(1, 3, figsize=(18, 6))

# Space 1: Pixel Space
ax1 = axes[0]
ax1.set_xlim(-1, 3)
ax1.set_ylim(-1, 3)
ax1.set_title('Pixel Space\n57,600 dimensions\n(240 × 240)', fontsize=12, weight='bold')
ax1.axis('off')

# Draw scattered points (high dimensional)
np.random.seed(42)
for _ in range(50):
    x, y = np.random.rand(2) * 2.5
    ax1.plot(x, y, 'o', color='lightgray', markersize=3)

# Draw anchors
for i in range(8):
    angle = i * 2 * np.pi / 8
    x, y = 1 + 0.6 * np.cos(angle), 1 + 0.6 * np.sin(angle)
    ax1.plot(x, y, 's', color='red', markersize=12, label='Anchor' if i == 0 else '')

ax1.text(1, -0.5, 'Raw images\nEigenface anchors\nVery high-dim', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat'))
ax1.legend(loc='upper right')

# Space 2: DINOv3 Space
ax2 = axes[1]
ax2.set_xlim(-1, 3)
ax2.set_ylim(-1, 3)
ax2.set_title('DINOv3 Embedding Space\n384 dimensions\n(pretrained features)', fontsize=12, weight='bold')
ax2.axis('off')

# Draw clustered points (semantic features)
for cluster in range(4):
    angle = cluster * np.pi / 2
    cx, cy = 1 + 0.5 * np.cos(angle), 1 + 0.5 * np.sin(angle)
    for _ in range(12):
        x, y = cx + 0.2 * np.random.randn(), cy + 0.2 * np.random.randn()
        ax2.plot(x, y, 'o', color='skyblue', markersize=5)

# Draw anchors
for i in range(8):
    angle = i * 2 * np.pi / 8
    x, y = 1 + 0.7 * np.cos(angle), 1 + 0.7 * np.sin(angle)
    ax2.plot(x, y, 's', color='orange', markersize=12)

ax2.text(1, -0.5, 'Semantic features\nAnchors projected\nFrozen during training', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat'))

# Arrow between spaces
fig2.text(0.365, 0.5, '→\nDINOv3\nBackbone\n(frozen)', ha='center', va='center', fontsize=11, weight='bold',
          bbox=dict(boxstyle='round', facecolor=color_backbone, edgecolor='black', linewidth=2))

# Space 3: Projected Space
ax3 = axes[2]
ax3.set_xlim(-1, 3)
ax3.set_ylim(-1, 3)
ax3.set_title('Projected Space\n128 dimensions\n(task-specific)', fontsize=12, weight='bold')
ax3.axis('off')

# Draw well-separated clusters
colors = plt.cm.Set3(np.linspace(0, 1, 8))
for cluster_idx in range(8):
    angle = cluster_idx * 2 * np.pi / 8
    cx, cy = 1 + 0.7 * np.cos(angle), 1 + 0.7 * np.sin(angle)
    
    # Samples close to anchor
    for _ in range(6):
        r = 0.15 * np.random.rand()
        theta = 2 * np.pi * np.random.rand()
        x, y = cx + r * np.cos(theta), cy + r * np.sin(theta)
        ax3.plot(x, y, 'o', color=colors[cluster_idx], markersize=6, alpha=0.6)
    
    # Anchor
    ax3.plot(cx, cy, 's', color=colors[cluster_idx], markersize=14, 
             markeredgecolor='black', markeredgewidth=2)
    ax3.text(cx, cy, str(cluster_idx+1), ha='center', va='center', 
             fontsize=8, weight='bold', color='white')

# Anomaly point
ax3.plot(1, 1, 'X', color='red', markersize=15, markeredgewidth=3, label='Anomaly\n(far from anchors)')

ax3.text(1, -0.5, 'Learned separation\nDistance comparisons\nAnomaly detection', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='wheat'))
ax3.legend(loc='upper right')

# Arrow between spaces
fig2.text(0.685, 0.5, '→\nProjection\nHead\n(trainable)', ha='center', va='center', fontsize=11, weight='bold',
          bbox=dict(boxstyle='round', facecolor=color_trainable, edgecolor='black', linewidth=2))

fig2.suptitle('Embedding Space Transformation: From Pixels to Task-Specific Features', 
              fontsize=14, weight='bold', y=0.98)

plt.tight_layout()
plt.savefig('embedding_spaces.png', dpi=300, bbox_inches='tight')
print("✅ Embedding spaces diagram saved: embedding_spaces.png")
plt.close()

print("\n" + "="*80)
print("DIAGRAMS GENERATED SUCCESSFULLY")
print("="*80)
print("\n1. architecture_diagram.png - Complete data flow with dimensions")
print("2. embedding_spaces.png - Transformation from pixel to projected space")
