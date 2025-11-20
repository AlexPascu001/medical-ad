"""
Test different anchor generation strategies
"""

import numpy as np
from anchors import AnchorGenerator
import matplotlib.pyplot as plt

# Create dummy images
print("Creating dummy training images...")
np.random.seed(42)
N, H, W = 100, 240, 240

# Generate some synthetic images with different patterns
images = []
for i in range(N):
    # Create images with various patterns
    if i < 30:
        # Bright center
        img = np.random.randn(H, W) * 0.1
        center = H // 2, W // 2
        y, x = np.ogrid[:H, :W]
        mask = (x - center[1])**2 + (y - center[0])**2 < (H//4)**2
        img[mask] += 0.5
    elif i < 60:
        # Vertical gradient
        img = np.linspace(0, 1, H)[:, None] + np.random.randn(H, W) * 0.1
    else:
        # Horizontal gradient
        img = np.linspace(0, 1, W)[None, :] + np.random.randn(H, W) * 0.1
    
    images.append(img)

images = np.array(images)
print(f"Generated {N} images of size {H}×{W}")

# Test all three strategies
strategies = ['random', 'kmeans', 'eigenface']
n_anchors = 8

fig, axes = plt.subplots(3, n_anchors, figsize=(n_anchors*2, 7))

for i, strategy in enumerate(strategies):
    print(f"\n{'='*80}")
    print(f"Testing {strategy.upper()} strategy")
    print(f"{'='*80}")
    
    # Create anchor generator
    anchor_gen = AnchorGenerator(
        strategy=strategy,
        n_components=20 if strategy == 'eigenface' else 50,
        n_anchors=n_anchors,
        random_state=42
    )
    
    # Generate anchors
    anchor_images = anchor_gen.fit(images)
    
    # Visualize
    for j in range(n_anchors):
        axes[i, j].imshow(anchor_images[j], cmap='gray')
        axes[i, j].axis('off')
        if j == 0:
            axes[i, j].set_ylabel(strategy.capitalize(), fontsize=12, rotation=0, 
                                 ha='right', va='center')
        if i == 0:
            axes[i, j].set_title(f'Anchor {j}', fontsize=10)

plt.tight_layout()
plt.savefig('test_anchor_strategies.png', dpi=150, bbox_inches='tight')
print(f"\n✓ Saved visualization to: test_anchor_strategies.png")

print(f"\n{'='*80}")
print("ALL STRATEGIES TESTED SUCCESSFULLY")
print(f"{'='*80}")
print(f"  ✓ Random: {n_anchors} anchors generated")
print(f"  ✓ K-means: {n_anchors} anchors generated")
print(f"  ✓ Eigenface: {n_anchors} anchors generated")
