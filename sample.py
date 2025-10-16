import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import timm
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image
import matplotlib.pyplot as plt
import shutil
import glob
from scipy.ndimage import zoom
import random

# --- Configuration ---
IMG_SIZE = 224
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 5
EMBEDDING_DIM = 128
K_EIGENFACES = 8
MARGIN = 1.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOAD_FROM_CHECKPOINT = False # Set to True to load a saved model
CHECKPOINT_PATH = "dino_ad_model.pth"
IOU_THRESHOLD = 0
print(f"Using device: {DEVICE}")

# --- Data Loading and Preprocessing ---
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(),
    transforms.ToTensor()
])

inv_normalize = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)

train_dataset = ImageFolder(root="data/BraTS2021_slice/train", transform=data_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --- Custom Dataset for Test data with masks ---
class AnomalyDataset(Dataset):
    def __init__(self, root_dir, transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, 'good', 'img', '*.png')))
        self.image_paths += sorted(glob.glob(os.path.join(root_dir, 'Ungood', 'img', '*.png')))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        label_str = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
        label = 0 if label_str == 'good' else 1

        mask = torch.zeros(1, IMG_SIZE, IMG_SIZE)
        if label == 1: # If 'Ungood', load the mask
            mask_path = img_path.replace('img', 'label')
            if os.path.exists(mask_path):
                mask_img = Image.open(mask_path)
                if self.mask_transform:
                    mask = self.mask_transform(mask_img)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label, mask

test_dataset = AnomalyDataset(root_dir="data/BraTS2021_slice/test", transform=data_transforms, mask_transform=mask_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# --- 4. Model Definition ---
class DinoAD(nn.Module):
    def __init__(self, embedding_dim):
        super(DinoAD, self).__init__()
        self.backbone = timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=True) 
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.backbone_features = self.backbone.num_features
        
        self.trainable_head = nn.Sequential(
            nn.Linear(self.backbone_features, self.backbone_features // 2),
            nn.ReLU(),
            nn.Linear(self.backbone_features // 2, embedding_dim)
        )

    def forward(self, x, return_patch_embeddings=False):
        features = self.backbone.forward_features(x)
        
        # Always calculate global embedding for consistency
        global_embedding = self.trainable_head(features.mean(dim=1))

        if return_patch_embeddings:
            # Exclude CLS token (at index 0) and register tokens
            patch_features = features[:, 5:, :] 
            patch_embeddings = self.trainable_head(patch_features)
            return patch_embeddings

        # Default behavior for training (image-level loss)
        return global_embedding

# --- 2. Calculate Feature-Space Anchors ---
def get_feature_anchors(dataset, model, k):
    print(f"Calculating top {k} feature-space anchors...")
    model.eval()
    all_features = []
    # Use a DataLoader for efficient processing
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    with torch.no_grad():
        for inputs, _ in loader:
            features = model.backbone.forward_features(inputs.to(DEVICE))
            # Use the global average of patch features as the representation
            global_features = features.mean(dim=1).cpu().numpy()
            all_features.append(global_features)
    
    data_matrix = np.vstack(all_features)
    
    pca = PCA(n_components=k)
    pca.fit(data_matrix)
    
    anchors = pca.components_
    print("Feature-space anchors calculated.")
    return torch.tensor(anchors, dtype=torch.float32).to(DEVICE)

# We need to instantiate the model first to use its backbone for anchor calculation
temp_model = DinoAD(embedding_dim=EMBEDDING_DIM).to(DEVICE)
feature_anchors = get_feature_anchors(train_dataset, temp_model, K_EIGENFACES)
del temp_model # Clean up the temporary model

def plot_eigenfaces(eigenfaces, img_size, num_to_show=K_EIGENFACES):
    """
    Visualizes the top eigenfaces.
    """
    # Make sure we don't try to show more eigenfaces than we have
    num_to_show = min(num_to_show, len(eigenfaces))
    
    # Create a figure to display the images
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    fig.suptitle('Top Eigenfaces', fontsize=16)

    for i in range(num_to_show):
        # Get the eigenface and reshape it
        eigenface = eigenfaces[i].cpu().numpy()
        eigenface_img = eigenface.reshape((img_size, img_size))
        
        # Plot it
        ax = axes[i//4, i%4]
        ax.imshow(eigenface_img, cmap='gray')
        ax.set_title(f"Eigenface {i+1}")
        ax.axis('off') # Hide the axes ticks
        
    plt.show()

# plot_eigenfaces(eigenface_anchors, IMG_SIZE)

# --- 3. Custom Loss Function ---
class CamLossForAD(nn.Module):
    def __init__(self, anchors, margin=1.0):
        super(CamLossForAD, self).__init__()
        self.register_buffer("anchors", anchors)
        self.margin = margin

    def forward(self, embeddings):
        # If we get patch embeddings, average them for the loss
        if embeddings.ndim == 3:
            embeddings = embeddings.mean(dim=1)
        dist_matrix = torch.cdist(embeddings, self.anchors)
        min_dists, _ = torch.min(dist_matrix, dim=1)
        attractor_loss = min_dists.pow(2).mean()
        
        return attractor_loss

# --- 5. Training ---
print("\n--- Starting Training ---")
model = DinoAD(embedding_dim=EMBEDDING_DIM).to(DEVICE)

if LOAD_FROM_CHECKPOINT and os.path.exists(CHECKPOINT_PATH):
    print(f"Loading model from {CHECKPOINT_PATH}")
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
else:
    print("Training new model...")
    # Project the feature-space anchors using the model's head and detach them
    projected_anchors = model.trainable_head(feature_anchors).detach()
    criterion = CamLossForAD(anchors=projected_anchors, margin=MARGIN).to(DEVICE)
    optimizer = optim.AdamW(model.trainable_head.parameters(), lr=LEARNING_RATE)

    model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for inputs, _ in train_loader:
            inputs = inputs.to(DEVICE)
            
            optimizer.zero_grad()
            embeddings = model(inputs)
            loss = criterion(embeddings)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss / len(train_loader):.4f}")

    print("Training finished.")
    print(f"Saving model to {CHECKPOINT_PATH}")
    torch.save(model.state_dict(), CHECKPOINT_PATH)

# --- 6. Evaluation and Results ---
def get_anomaly_map(model, patch_embeddings, projected_anchors):
    # patch_embeddings shape: (batch_size, num_patches, embedding_dim)
    dist_matrix = torch.cdist(patch_embeddings, projected_anchors)
    min_dists, _ = torch.min(dist_matrix, dim=-1) # Shape: (batch_size, num_patches)
    
    # Reshape to a 2D map for each image in the batch
    patch_size = model.backbone.patch_embed.patch_size[0]
    map_size = IMG_SIZE // patch_size
    anomaly_map = min_dists.reshape(-1, map_size, map_size) # Shape: (batch_size, map_size, map_size)

    # Upscale to image size
    upscaled_map = zoom(anomaly_map.cpu().numpy(), (1, patch_size, patch_size), order=1)
    return torch.tensor(upscaled_map)

def calculate_iou(pred_mask, true_mask):
    # The threshold is now applied outside this function
    pred_mask = pred_mask.int()
    true_mask = (true_mask > 0).int()
    
    intersection = torch.sum(pred_mask & true_mask)
    union = torch.sum(pred_mask | true_mask)
    
    iou = (intersection + 1e-6) / (union + 1e-6) # Add epsilon for stability
    return iou.item()

print("\n--- Evaluating Model ---")
model.eval()
# Project the feature-space anchors using the final trained model's head
projected_anchors = model.trainable_head(feature_anchors)
results = []
total_iou = 0
anomaly_count = 0

with torch.no_grad():
    for (inputs, labels, masks) in test_loader:
        patch_embeddings = model(inputs.to(DEVICE), return_patch_embeddings=True)
        
        anomaly_maps = get_anomaly_map(model, patch_embeddings, projected_anchors)
        
        for i in range(inputs.size(0)):
            is_anomaly = labels[i].item() == 1
            
            map_score = anomaly_maps[i].max().item()
            
            # IOU will be calculated later with the dynamic threshold
            iou = 0 
            if is_anomaly:
                # We still calculate a temporary IoU here for average reporting if needed,
                # but the one for display will be recalculated.
                temp_pred_mask = (anomaly_maps[i] > 0.5).int() # Using a placeholder threshold
                iou = calculate_iou(temp_pred_mask, masks[i])
                total_iou += iou
                anomaly_count += 1

            results.append({
                "image": inputs[i].cpu(),
                "label": "Ungood" if is_anomaly else "good",
                "map_score": map_score,
                "iou": iou,
                "anomaly_map": anomaly_maps[i].cpu(),
                "true_mask": masks[i].cpu()
            })

max_score = max(r['map_score'] for r in results)
print(f"\nMax anomaly map score in test set: {max_score:.4f}")

if anomaly_count > 0:
    print(f"\nAverage IoU on anomalous samples (using placeholder threshold): {total_iou / anomaly_count:.4f}")

# --- Determine Dynamic Threshold ---
good_scores = [r['map_score'] for r in results if r['label'] == 'good']
if good_scores:
    IOU_THRESHOLD = np.percentile(good_scores, 95)
    print(f"Using dynamic IoU threshold (95th percentile of 'good' scores): {IOU_THRESHOLD:.4f}")
else:
    # Fallback if there are no 'good' samples in the test set
    IOU_THRESHOLD = np.percentile([r['map_score'] for r in results], 90)
    print(f"Warning: No 'good' samples in test set. Using 90th percentile of all scores as threshold: {IOU_THRESHOLD:.4f}")


# --- Display Results ---
print("\n--- Anomaly Detection Results ---")
# Show 3 random 'good' and 3 random 'Ungood' samples
good_samples = [r for r in results if r['label'] == 'good']
ungood_samples = [r for r in results if r['label'] == 'Ungood']
samples_to_show = random.sample(good_samples, 3) + random.sample(ungood_samples, 3)

fig, axs = plt.subplots(len(samples_to_show), 4, figsize=(16, 4 * len(samples_to_show)))
fig.suptitle("Model Predictions on Test Set", fontsize=16)


for i, res in enumerate(samples_to_show):
    # 1. Original Image
    ax = axs[i, 0]
    img = inv_normalize(res['image']).permute(1, 2, 0).numpy()
    ax.imshow(np.clip(img, 0, 1))
    title = f"True: {res['label']} | Map Score: {res['map_score']:.2f}"
    ax.set_title(title)
    ax.axis('off')

    # 2. True Mask
    ax = axs[i, 1]
    ax.imshow(res['true_mask'].squeeze(), cmap='gray')
    ax.set_title(f"Ground Truth")
    ax.axis('off')

    # 3. Predicted Mask (using the new dynamic threshold)
    ax = axs[i, 2]
    pred_mask = (res['anomaly_map'] > IOU_THRESHOLD).int()
    # Recalculate IoU with the correct threshold for display
    final_iou = calculate_iou(pred_mask, res['true_mask'])
    ax.imshow(pred_mask.squeeze(), cmap='gray')
    ax.set_title(f"Predicted Mask | (IoU: {final_iou:.2f})")
    ax.axis('off')

    # 4. Predicted Anomaly Map (Heatmap)
    ax = axs[i, 3]
    pred_map_plot = ax.imshow(res['anomaly_map'].squeeze(), cmap='jet', vmin=0, vmax=max_score)
    ax.set_title(f"Heatmap")
    ax.axis('off')
    fig.colorbar(pred_map_plot, ax=ax, fraction=0.046, pad=0.04)


plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# --- Cleanup ---
# shutil.rmtree("medical_dataset")
