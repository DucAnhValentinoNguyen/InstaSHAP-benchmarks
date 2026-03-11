import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset, Subset
from datasets import load_dataset, concatenate_datasets
from sklearn.model_selection import KFold
import numpy as np
import os
import matplotlib.pyplot as plt
import time

# --- 1. RTX 4090 MAX PERFORMANCE SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

BATCH_SIZE = 768 # Maximizing the 24GB VRAM
EPOCHS = 35       # Increased for interaction heads to converge
K_FOLDS = 5
NUM_WORKERS = 8  # Maximizing CPU-to-GPU throughput

# --- 2. DATASET PREPARATION ---
print(">>> Loading Tiny-ImageNet...")
train_hf = load_dataset("zh-plus/tiny-imagenet", split="train") 
val_hf = load_dataset("zh-plus/tiny-imagenet", split="valid")
full_dataset_hf = concatenate_datasets([train_hf, val_hf])

# Using TrivialAugmentWide for better generalization at 35 epochs
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(), 
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class CVDataset(Dataset):
    def __init__(self, hf_data, transform):
        self.hf_data = hf_data
        self.transform = transform
    def __len__(self): return len(self.hf_data)
    def __getitem__(self, idx):
        item = self.hf_data[idx]
        return self.transform(item["image"]), item["label"]

# --- 3. GAM-3 ARCHITECTURE (Factorized) ---
class InstaSHAP_GAM3(nn.Module):
    def __init__(self, num_patches=49, embed_dim=128):
        super().__init__()
        res50 = models.resnet50(weights='IMAGENET1K_V2')
        self.backbone = nn.Sequential(*list(res50.children())[:-2])
        
        # Unfreeze Layer 3 and Layer 4 for higher capacity!
        for p in self.backbone.parameters(): p.requires_grad = False
        for p in self.backbone[6].parameters(): p.requires_grad = True # Layer 3
        for p in self.backbone[7].parameters(): p.requires_grad = True # Layer 4
        
        self.embedding_proj = nn.Linear(2048, embed_dim)
        self.norm = nn.LayerNorm(embed_dim) # Crucial for 3rd-order stability
        
        self.head_order1 = nn.Linear(embed_dim, 200)
        self.head_order2 = nn.Linear(embed_dim, 200)
        self.head_order3 = nn.Linear(embed_dim, 200)

    def forward(self, x, mask, return_components=False):
        feat = self.backbone(x) 
        B, C, H, W = feat.shape
        
        pts = feat.view(B, C, H*W).permute(0, 2, 1)
        h = self.norm(self.embedding_proj(pts))
        h_masked = h * mask.unsqueeze(-1)
        
        # --- Order 1 (Main Effects) ---
        out1_spatial = self.head_order1(h_masked) 
        out1 = out1_spatial.sum(dim=1)
        
        # --- Order 2 (2-way Interactions) ---
        sum_h = torch.sum(h_masked, dim=1)
        sum_h_sq = torch.sum(h_masked**2, dim=1)
        inter2 = 0.5 * (sum_h**2 - sum_h_sq)
        out2 = self.head_order2(inter2)
        
        # --- Order 3 (3-way Interactions) ---
        sum_h_cub = torch.sum(h_masked**3, dim=1)
        inter3 = (1/6) * (sum_h**3 - 3 * sum_h * sum_h_sq + 2 * sum_h_cub)
        out3 = self.head_order3(inter3)
        
        logits = out1 + out2 + out3
        
        if return_components:
            return logits, out1_spatial, out2, out3
        return logits

# --- 4. TESTS & METRICS ---
def faithfulness_deletion_test(model, val_loader):
    model.eval()
    mask_fractions = [0.0, 0.1, 0.3, 0.5, 0.7]
    confidence_drops = {frac: [] for frac in mask_fractions}
    
    print("\n>>> Running Faithfulness Test on GAM-3...")
    with torch.no_grad():
        imgs, labels = next(iter(val_loader))
        imgs, labels = imgs.to(device), labels.to(device)
        
        full_mask = torch.ones((imgs.size(0), 49), device=device)
        base_logits, spatial_shap, _, _ = model(imgs, full_mask, return_components=True)
        base_probs = F.softmax(base_logits, dim=1)
        base_true_probs = base_probs[torch.arange(imgs.size(0)), labels]
        
        for i in range(imgs.size(0)):
            # Use Order-1 main effects to rank patch importance
            img_shap = spatial_shap[i, :, labels[i]] 
            sorted_indices = torch.argsort(img_shap, descending=True)
            
            for frac in mask_fractions:
                num_to_drop = int(frac * 49)
                test_mask = torch.ones(49, device=device)
                if num_to_drop > 0:
                    test_mask[sorted_indices[:num_to_drop]] = 0.0
                    
                test_logit = model(imgs[i:i+1], test_mask.unsqueeze(0)) 
                test_prob = F.softmax(test_logit, dim=1)[0, labels[i]]
                confidence_drops[frac].append((base_true_probs[i] - test_prob).item())
    
    print("[Faithfulness] GAM-3 Masking Top-k% features -> Avg Confidence Drop")
    for frac in mask_fractions:
        print(f"  Drop {frac*100:2.0f}% patches : {np.mean(confidence_drops[frac]):+.4f}")

def consistency_test(model_paths, val_loader):
    print("\n>>> Running Consistency Test across GAM-3 Ensemble...")
    models = []
    for path in model_paths:
        m = InstaSHAP_GAM3().to(device)
        state_dict = torch.load(path)
        clean_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
        m.load_state_dict(clean_dict)
        m.eval()
        models.append(m)

    with torch.no_grad():
        imgs, labels = next(iter(val_loader))
        imgs, labels = imgs.to(device), labels.to(device)
        full_mask = torch.ones((imgs.size(0), 49), device=device)
        
        ensemble_shaps = [] 
        for m in models:
            _, spatial_shap, _, _ = m(imgs, full_mask, return_components=True)
            true_class_shaps = spatial_shap[torch.arange(imgs.size(0)), :, labels]
            ensemble_shaps.append(true_class_shaps)
            
        ensemble_shaps = torch.stack(ensemble_shaps)
        mean_std = torch.mean(torch.std(ensemble_shaps, dim=0)).item()
        
    print(f"[Consistency] Average Main-Effect SHAP Std Dev across 5 folds: {mean_std:.4f}")

def visualize_gam3_components(model_paths, image_tensor, original_image, target_class=None):
    """Visualizes the Main Effects and Interaction Contributions of the GAM-3."""
    image_tensor = image_tensor.to(device)
    mask = torch.ones((1, 49), device=device)
    
    m = InstaSHAP_GAM3().to(device)
    state_dict = torch.load(model_paths[0])
    m.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state_dict.items()})
    m.eval()
    
    with torch.no_grad():
        logits, out1_spatial, out2, out3 = m(image_tensor, mask, return_components=True)
        if target_class is None: target_class = logits.argmax(dim=1).item()
        
        main_effect_shap = out1_spatial[0, :, target_class].cpu().view(7, 7)
        o1_val = main_effect_shap.sum().item()
        o2_val = out2[0, target_class].item()
        o3_val = out3[0, target_class].item()

    upsampled_main = F.interpolate(main_effect_shap.unsqueeze(0).unsqueeze(0), 
                                   size=(224, 224), mode='bicubic').squeeze().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    vmax = np.max(np.abs(upsampled_main))
    im = axes[1].imshow(original_image, alpha=0.4)
    im = axes[1].imshow(upsampled_main, cmap='coolwarm', vmin=-vmax, vmax=vmax, alpha=0.7)
    axes[1].set_title(f"GAM-3 Main Effects (Class {target_class})")
    plt.colorbar(im, ax=axes[1])
    axes[1].axis('off')
    
    print("\n--- GAM-3 Prediction Decomposition ---")
    print(f"Total Logit for Class {target_class}: {logits[0, target_class].item():.4f}")
    print(f"  ├─ Order 1 (Main Effects) : {o1_val:+.4f}")
    print(f"  ├─ Order 2 (Pairs)        : {o2_val:+.4f}")
    print(f"  └─ Order 3 (Triplets)     : {o3_val:+.4f}")
    
    plt.tight_layout()
    plt.show()

# --- 5. MAIN TRAINING LOOP ---
if __name__ == "__main__":
    os.makedirs("gam3_checkpoints", exist_ok=True)
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    saved_models = []

    print(f">>> Starting GAM-3 {K_FOLDS}-Fold Training (Targeting High Accuracy)...")
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(full_dataset_hf)))):
        print(f"\n{'='*20} Fold {fold + 1} {'='*20}")
        
        train_loader = DataLoader(Subset(CVDataset(full_dataset_hf, train_transform), train_idx), 
                                  batch_size=BATCH_SIZE, shuffle=True, 
                                  num_workers=NUM_WORKERS, pin_memory=True) # Removed persistent_workers
        val_loader = DataLoader(Subset(CVDataset(full_dataset_hf, val_transform), val_idx), 
                                batch_size=BATCH_SIZE, shuffle=False, 
                                num_workers=NUM_WORKERS, pin_memory=True) # Removed persistent_workers
        
        model = InstaSHAP_GAM3().to(device)
        compiled_model = torch.compile(model)
        
        # optimizer = optim.AdamW([
        #     {'params': compiled_model.backbone[6].parameters(), 'lr': 5e-6}, # Layer 3
        #     {'params': compiled_model.backbone[7].parameters(), 'lr': 1e-5}, # Layer 4
        #     {'params': compiled_model.embedding_proj.parameters(), 'lr': 1e-3},
        #     {'params': compiled_model.norm.parameters(), 'lr': 1e-3},
        #     {'params': compiled_model.head_order1.parameters(), 'lr': 1e-3},
        #     {'params': compiled_model.head_order2.parameters(), 'lr': 1e-3},
        #     {'params': compiled_model.head_order3.parameters(), 'lr': 1e-3}
        # ], fused=True)


        # optimizer = optim.AdamW([
        #     # Backbone: Slow updates to preserve ImageNet features
        #     {'params': compiled_model.backbone[6].parameters(), 'lr': 1e-5}, 
        #     {'params': compiled_model.backbone[7].parameters(), 'lr': 1e-5},
        #     # Heads: Fast updates to learn the Tiny-ImageNet mappings
        #     {'params': compiled_model.embedding_proj.parameters(), 'lr': 1e-3},
        #     {'params': compiled_model.norm.parameters(), 'lr': 1e-3},
        #     {'params': compiled_model.head_order1.parameters(), 'lr': 1e-3},
        #     {'params': compiled_model.head_order2.parameters(), 'lr': 1e-3},
        #     {'params': compiled_model.head_order3.parameters(), 'lr': 1e-3}
        # ], weight_decay=0.01, fused=True)

        # Separate the parameters into two groups
        # optimizer = optim.AdamW([
        #     # Group 1: Pre-trained Backbone (Keep it slow/conservative)
        #     {'params': compiled_model.backbone[6].parameters(), 'lr': 1e-5}, 
        #     {'params': compiled_model.backbone[7].parameters(), 'lr': 1e-5},
            
        #     # Group 2: New GAM-3 Interaction Heads (Aggressive learning)
        #     {'params': compiled_model.embedding_proj.parameters(), 'lr': 1e-3},
        #     {'params': compiled_model.norm.parameters(), 'lr': 1e-3},
        #     {'params': compiled_model.head_order1.parameters(), 'lr': 1e-3},
        #     {'params': compiled_model.head_order2.parameters(), 'lr': 1e-3},
        #     {'params': compiled_model.head_order3.parameters(), 'lr': 1e-3}
        # ], weight_decay=0.01, fused=True)

        

        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        # 1. Update the Optimizer to use Differential Learning Rates
        # 1. DIFFERENTIAL LEARNING RATES
        # optimizer = optim.AdamW([
        #     {'params': compiled_model.backbone[6].parameters(), 'lr': 1e-5}, # Layer 3
        #     {'params': compiled_model.backbone[7].parameters(), 'lr': 1e-5}, # Layer 4
        #     {'params': compiled_model.embedding_proj.parameters(), 'lr': 1e-3},
        #     {'params': compiled_model.norm.parameters(), 'lr': 1e-3},
        #     {'params': compiled_model.head_order1.parameters(), 'lr': 1e-3},
        #     {'params': compiled_model.head_order2.parameters(), 'lr': 1e-3},
        #     {'params': compiled_model.head_order3.parameters(), 'lr': 1e-3}
        # ], weight_decay=0.01, fused=True)


        optimizer = optim.AdamW([
            # UNFREEZE ENTIRE BACKBONE: Let the features adapt to the interactions
            {'params': compiled_model.backbone.parameters(), 'lr': 2e-5}, # Slightly higher LR
            
            # Interaction Heads
            {'params': compiled_model.embedding_proj.parameters(), 'lr': 1e-3},
            {'params': compiled_model.norm.parameters(), 'lr': 1e-3},
            {'params': compiled_model.head_order1.parameters(), 'lr': 1e-3},
            {'params': compiled_model.head_order2.parameters(), 'lr': 1e-3},
            {'params': compiled_model.head_order3.parameters(), 'lr': 1e-3}
        ], weight_decay=0.01, fused=True)

        # 2. ONECYCLE WARMUP SCHEDULER
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=[1e-5, 1e-5, 1e-3, 1e-3, 1e-3, 1e-3, 1e-3], 
            epochs=EPOCHS, 
            steps_per_epoch=len(train_loader),
            pct_start=0.3 # Spend the first 30% of training warming up
        )
        
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0
        model_path = f"gam3_checkpoints/instashap_gam3_fold_{fold+1}.pth"

        for epoch in range(EPOCHS):
            compiled_model.train()
            start_time = time.time()
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                mask = torch.bernoulli(torch.full((imgs.size(0), 49), 0.5, device=device))
                
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = criterion(compiled_model(imgs, mask), labels)
                loss.backward()
                optimizer.step()
                
                # 3. CRITICAL: OneCycleLR steps PER BATCH, not per epoch!
                scheduler.step() 
            
            compiled_model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    mask = torch.ones((imgs.size(0), 49), device=device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        logits = compiled_model(imgs, mask)
                    correct += (torch.argmax(logits, dim=1) == labels).sum().item()
                    total += labels.size(0)

            acc = (correct / total) * 100
            epoch_time = time.time() - start_time
            
            # Print the Head LR (index 2) to watch the warmup happen
            current_head_lr = scheduler.get_last_lr()[2] 
            print(f"  Epoch {epoch+1:02d} | Head LR: {current_head_lr:.6f} | Val Acc: {acc:.2f}% | Time: {epoch_time:.1f}s")
            
            if acc > best_acc:
                best_acc = acc
                torch.save(compiled_model.state_dict(), model_path)
                
        saved_models.append(model_path)

    # --- 6. RUN EXPERIMENTS ---
    print("\n" + "="*40)
    print(">>> RUNNING GAM-3 A* EXPERIMENTS")
    print("="*40)
    
    # We use a standard data loader without persistent workers for the quick eval script
    test_loader = DataLoader(Subset(CVDataset(full_dataset_hf, val_transform), val_idx), 
                             batch_size=128, shuffle=True, num_workers=4)
    
    # 1. Faithfulness Test
    best_fold1 = InstaSHAP_GAM3().to(device)
    best_fold1.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in torch.load(saved_models[0]).items()})
    faithfulness_deletion_test(best_fold1, test_loader)
    
    # 2. Consistency Test
    consistency_test(saved_models, test_loader)
    
    # 3. Visualization
    sample_img, sample_label = next(iter(test_loader))
    orig_img = sample_img[0].permute(1, 2, 0).numpy()
    orig_img = (orig_img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
    orig_img = np.clip(orig_img, 0, 1)
    
    visualize_gam3_components(saved_models, sample_img[0:1], orig_img, target_class=sample_label[0].item())