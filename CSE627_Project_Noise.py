import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
# Path to save loss history
LOSS_HISTORY_PATH = 'loss_history.npy'
from tqdm import tqdm

# ---------- Configuration ----------
USE_SAVED_MODEL = False   # Set to True to skip training and load saved weights
DEBUG = True        # Set to False for full training
VAL_SPLIT = 0.2            # fraction of data for validation
LOSS_METHOD = 'BCE'        # options: 'BCE' for BCEWithLogitsLoss, 'MSE' for MSELoss
MODEL_SAVE_PATH = 'best_unet.pth'
PRINT_LOSS_EVERY = 5       # log interval for full mode
ML_METHOD = 'iterative'    # options: 'baseline','iterative','multi','misaligned'
USE_NOISY_DATA = True   # Set to False to use clear data instead
GENERATE_IMG = True    # Determine if generate new images base on saved param
NODE_EVAL = False        # True - use evaluator/recall, False - no evaluator/recall
max_eval = 20          # Do evaluator/recall for the first max_eval images
# ---------- Dataset Definition ----------
class RoadSkeletonDataset(Dataset):
    def saveInput(self, tensor, img_name):
        """
        Save the transformed (blurred, noisy) input tensor as a PNG in save_dir.
        """
        pil = T.ToPILImage()(tensor.squeeze(0))
        pil.save(os.path.join(self.save_dir, img_name))

    """
    PyTorch Dataset for loading road skeleton images.
    Expects files named image_XXXXX.png and target_XXXXX.png in root_dir.
    Also saves each blurred/noisy input to 'saved_bluured_inputs'.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = sorted([
            f for f in os.listdir(root_dir)
            if f.startswith("image_") and f.lower().endswith(".png")
        ])
        # Base transform (for saved inputs)
        self.base_transform = T.Compose([
            T.Grayscale(),
            T.ToTensor(),
        ])
        # Noise/blur transform
        self.noise_transform = T.Compose([
            T.Grayscale(),
            T.ToTensor(),
            T.Lambda(lambda t: t + 0.05 * torch.randn_like(t)),
            T.RandomApply([T.GaussianBlur(kernel_size=5)], p=0.5),
            T.Lambda(lambda t: torch.clamp(t, 0.0, 1.0)),
        ])
        # Directory to save blurred/noisy inputs alongside the script
        self.save_dir = os.path.join(os.getcwd(), 'saved_bluured_inputs')
        os.makedirs(self.save_dir, exist_ok=True)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        target_name = img_name.replace("image_", "target_")

        # Choose clear or noisy input based on global flag
        if USE_NOISY_DATA:
            # Load or generate noisy input for this index
            saved_path = os.path.join(self.save_dir, img_name)
            if os.path.exists(saved_path):
                img = Image.open(saved_path)
                img_t = self.base_transform(img)
            else:
                img_orig = Image.open(os.path.join(self.root_dir, img_name))
                img_t = self.noise_transform(img_orig)
                # Save this noisy example for future reuse
                self.saveInput(img_t, img_name)

        # Load and transform target (always clear skeleton)
        target = Image.open(os.path.join(self.root_dir, target_name))
        target_t = self.base_transform(target)
        return img_t, target_t

# ---------- U-Net Model Definitions ----------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )
    def forward(self, x): return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        return self.conv(torch.cat([x2, x1], dim=1))

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch): super().__init__(); self.conv = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super().__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024//factor)
        self.up1 = Up(1024, 512//factor, bilinear)
        self.up2 = Up(512, 256//factor, bilinear)
        self.up3 = Up(256, 128//factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

class SimpleUNet(UNet):
    def __init__(self, n_channels=1, n_classes=1):
        super().__init__(n_channels, n_classes)
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.up1 = Up(32+16, 16)
        self.outc = OutConv(16, n_classes)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x = self.up1(x2, x1)
        return self.outc(x)

class IterUnet(UNet):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True, iterations=3):
        super().__init__(n_channels, n_classes, bilinear)
        self.iterations = iterations
    def forward(self, x):
        logits = None
        for _ in range(self.iterations):
            logits = super().forward(x)
            prob = torch.sigmoid(logits)
            x = x * prob
        return logits

class MultiUnet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super().__init__()
        self.unet = UNet(n_channels, n_classes, bilinear)
        self.dist_head = nn.Conv2d(n_classes, 1, 1)
    def forward(self, x):
        logits = self.unet(x)
        prob = torch.sigmoid(logits)
        dist = self.dist_head(prob)
        return logits, dist

class MisalignUnet(UNet):
    def __init__(self, n_channels=1, n_classes=1, bilinear=True):
        super().__init__(n_channels, n_classes, bilinear)
        self.localization = nn.Sequential(
            nn.Conv2d(1,8,7), nn.MaxPool2d(2), nn.ReLU(True),
            nn.Conv2d(8,10,5), nn.MaxPool2d(2), nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(nn.Linear(10*64*64,32), nn.ReLU(True), nn.Linear(32,6))
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0],dtype=torch.float))
    def stn(self,x):
        xs = self.localization(x); xs=xs.view(xs.size(0),-1)
        theta = self.fc_loc(xs).view(-1,2,3)
        grid = F.affine_grid(theta,x.size(),align_corners=True)
        return F.grid_sample(x,grid,align_corners=True)
    def forward(self,x): return super().forward(self.stn(x))

# ---------- Node Precision & Recall Evaluation ----------
class NodeEvaluator:
    def __init__(self, gt_dir, pred_dir, max_dist=3):
        self.gt_dir = gt_dir
        self.pred_dir = pred_dir
        self.max_dist = max_dist

    def get_nodes(self, mask):
        skel = skeletonize(mask)
        coords = np.column_stack(np.nonzero(skel))
        valence_dict = {}
        for r, c in coords:
            neighbors = skel[max(r-1, 0):r+2, max(c-1, 0):c+2].sum() - 1
            valence_dict.setdefault(neighbors, []).append((r, c))
        return valence_dict

    def evaluate(self):
        # Prepare list of valid ground-truth files
        files = [f for f in os.listdir(self.gt_dir)
                 if f.startswith('target_') and f.lower().endswith('.png')]
        # Take only up to max_eval entries
        files = files[:max_eval]
        results = {k: {'gt':0, 'pred':0, 'match':0} for k in [1,2,3,4]}
        # Iterate with a progress bar sized to max_eval
        for fname in tqdm(files, desc='Node Eval', unit='img', total=len(files)):
            # process only first max_eval images
            # Ground-truth mask
            gt_mask = np.array(
                Image.open(os.path.join(self.gt_dir, fname)).convert('L')
            ) > 0
            image_name = fname.replace('target_', 'image_')
            pred_name = f'pred_{image_name}'
            pred_path = os.path.join(self.pred_dir, pred_name)
            if not os.path.exists(pred_path):
                continue
            # Prediction mask
            pred_mask = np.array(
                Image.open(pred_path).convert('L')
            ) > 0
            gt_nodes = self.get_nodes(gt_mask)
            pred_nodes = self.get_nodes(pred_mask)
            for k in results:
                Gk = gt_nodes.get(k, [])
                Pk = pred_nodes.get(k, [])
                results[k]['gt'] += len(Gk)
                results[k]['pred'] += len(Pk)
                if not Gk or not Pk:
                    continue
                dist = cdist(Gk, Pk)
                B = nx.Graph()
                # ensure all nodes are present in the bipartite graph
                left_nodes = [('g', i) for i in range(len(Gk))]
                right_nodes = [('p', j) for j in range(len(Pk))]
                B.add_nodes_from(left_nodes, bipartite=0)
                B.add_nodes_from(right_nodes, bipartite=1)
                for i in range(len(Gk)):
                    for j in range(len(Pk)):
                        if dist[i, j] <= self.max_dist:
                            B.add_edge(('g', i), ('p', j))
                matching = nx.algorithms.bipartite.matching.hopcroft_karp_matching(
                    B, top_nodes=[('g', i) for i in range(len(Gk))]
                )
                match_count = sum(
                    1 for u in matching if isinstance(u, tuple) and u[0][0] == 'g'
                )
                results[k]['match'] += match_count
        for k, vals in results.items():
            gt, pr, m = vals['gt'], vals['pred'], vals['match']
            prec = m/pr if pr>0 else 0
            rec = m/gt if gt>0 else 0
            print(f"Valence {k}: Precision={prec:.3f} ({m}/{pr}), Recall={rec:.3f} ({m}/{gt})")

# ---------- Imports for Node Evaluation ---------- ----------
from skimage.morphology import skeletonize
import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
# ---------- Training Loop with Model Selection ----------
def train_model(data_dir, epochs=20, batch_size=8, lr=1e-3):
    # Track loss history for plotting
    loss_history = []

    # Skip training if loading saved model
    if USE_SAVED_MODEL and os.path.exists(MODEL_SAVE_PATH):
        print("Skipping training. Using saved weights.")
        return

    print('Debug Mode' if DEBUG else 'Full Training Mode')
    print(f"ML method: {ML_METHOD}, Loss: {LOSS_METHOD}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = RoadSkeletonDataset(data_dir)
    if DEBUG:
        ds = torch.utils.data.Subset(ds, list(range(min(20, len(ds)))))
        print(f"[DEBUG] subset size: {len(ds)}")
    val_n = int(len(ds) * VAL_SPLIT) if not DEBUG else 0
    train_n = len(ds) - val_n
    if val_n:
        train_ds, val_ds = random_split(ds, [train_n, val_n])
    else:
        train_ds, val_ds = ds, None
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size) if val_ds else None

    # Model selection
    if DEBUG:
        model = SimpleUNet()
    else:
        if ML_METHOD == 'iterative': model = IterUnet()
        elif ML_METHOD == 'multi': model = MultiUnet()
        elif ML_METHOD == 'misaligned': model = MisalignUnet()
        else: model = UNet()
    model.to(device)

    # Loss criterion
    if LOSS_METHOD == 'BCE':
        criterion = nn.BCEWithLogitsLoss()
    elif LOSS_METHOD == 'MSE':
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unsupported LOSS_METHOD: {LOSS_METHOD}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = float('inf')

    # Training loop
    for ep in tqdm(range(1, epochs + 1), desc='Epoch', unit='ep'):
        model.train()
        running_loss = 0.0
        correct = total = 0
        for imgs, tg in tqdm(train_loader, desc=f'Ep {ep}/{epochs}', unit='b'):
            imgs, tg = imgs.to(device), tg.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            logits = out[0] if isinstance(out, tuple) else out
            loss = criterion(logits, tg)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == (tg > 0.5)).sum().item()
            total += preds.numel()
        epoch_loss = running_loss / train_n
        loss_history.append(epoch_loss)
        epoch_acc = correct / total

        # Checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)

        # Logging
        if DEBUG or ep % PRINT_LOSS_EVERY == 0:
            log = f"Ep {ep}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}"
            if val_loader:
                model.eval()
                v_correct = v_total = 0
                with torch.no_grad():
                    for imgs, tg in val_loader:
                        imgs, tg = imgs.to(device), tg.to(device)
                        pr = (torch.sigmoid(model(imgs) if not isinstance(model(imgs), tuple) else model(imgs)[0]) > 0.5).float()
                        v_correct += (pr == (tg > 0.5)).sum().item()
                        v_total += pr.numel()
                val_acc = v_correct / v_total
                log += f", Val Acc={val_acc:.4f}"
            print(log)

    # Save loss history for plotting
    np.save(LOSS_HISTORY_PATH, np.array(loss_history))
    print(f"Saved loss history to {LOSS_HISTORY_PATH}")

# ---------- End of train_model ----------

if __name__=='__main__':
    base=os.path.dirname(os.path.abspath(__file__))
    data_dir=os.path.join(base,'thinning_data','data','thinning')
    print('Loading data from:',data_dir)
    # Prepare dataset and (optionally) save noisy inputs
    ds = RoadSkeletonDataset(data_dir)
    if USE_NOISY_DATA:
        if os.path.exists(ds.save_dir): 
            print('using existing noisy data')
        else:
            print('generate new noisy data and saving it %d', ds.save_dir)
        
        for i in range(len(ds)):
            _img, _ = ds[i]
        print('Saved to', ds.save_dir)
    else:
        print('Using clear data mode; no noisy inputs will be generated')
    # Train
    train_model(data_dir)
    # Plot loss history after training, even in debug mode
    if os.path.exists(LOSS_HISTORY_PATH):
        losses = np.load(LOSS_HISTORY_PATH)
        plt.figure()
        plt.plot(np.arange(1, len(losses)+1), losses, marker='o')
        plt.title('Training Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('loss_plot.png')
        print('Saved loss plot to loss_plot.png')
    # Inference
    print('Inference...')
    ds=RoadSkeletonDataset(data_dir); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if DEBUG: model=SimpleUNet(); print('Loading SimpleUNet (debug)')
    else:
        if ML_METHOD=='iterative': model=IterUnet()
        elif ML_METHOD=='multi': model=MultiUnet()
        elif ML_METHOD=='misaligned': model=MisalignUnet()
        else: model=UNet()
        print(f"Loading {model.__class__.__name__} for inference")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH,map_location=device))
    model.to(device); model.eval()
    pred_dir=os.path.join(base,'predictions'); os.makedirs(pred_dir,exist_ok=True)

    # Evaluate Node Precision & Recall
    if NODE_EVAL:
        evaluator = NodeEvaluator(data_dir, pred_dir)
        evaluator.evaluate()
    
    if GENERATE_IMG:
        with torch.no_grad():
            # Progress counter for Pic generation
            for img_name in tqdm(RoadSkeletonDataset(data_dir).image_files, desc='Generating Pictures', unit='img'):
                img = Image.open(os.path.join(data_dir, img_name)).convert('L')
                t = T.ToTensor()(img).unsqueeze(0).to(device)
                out = model(t)
                logits = out[0] if isinstance(out, tuple) else out
                prob = torch.sigmoid(logits)[0,0]
                mask = (prob > 0.45).cpu().numpy().astype('uint8') * 255
                soft = (prob * 255).cpu().numpy().astype('uint8')
                Image.fromarray(mask).save(os.path.join(pred_dir, f"pred_{img_name}"))
                Image.fromarray(soft).save(os.path.join(pred_dir, f"soft_{img_name}"))
                # tqdm will handle progress display
                #print('Saved', img_name)
            #break
                #break

