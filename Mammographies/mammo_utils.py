import os
import sys
import random
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import glob
# from glob import glob
from tqdm import tqdm
import json
from torch.cuda.amp import autocast, GradScaler
import time
from collections import Counter
# skimage / filters / morphology / io / color / measure
from skimage import filters, morphology
from skimage.filters import threshold_otsu, gaussian
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from skimage.measure import label as sk_label
from sklearn.model_selection import train_test_split
# scipy.ndimage helpers (label alias kept as nd_label where used)
import math
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.ndimage import (binary_fill_holes, gaussian_filter1d,
                           binary_erosion, binary_dilation, label as nd_label)
# torch / torchvision (used in training cells)
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler
from torchvision import models, transforms
import torch.nn.functional as F
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")



# Utilitaire: crop des bords (30 px par défaut)
def crop_borders(img, pad=30):
    if img is None:
        return None
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            h, w = img.shape
            if h <= 2*pad or w <= 2*pad:
                return img.copy()
            return img[pad:h-pad, pad:w-pad]
        elif img.ndim == 3:
            h, w, _ = img.shape
            if h <= 2*pad or w <= 2*pad:
                return img.copy()
            return img[pad:h-pad, pad:w-pad, :]
    # Fallback pour objets type PIL
    try:
        arr = np.array(img)
    except Exception:
        return img
    return crop_borders(arr, pad)



def get_mask_of_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    lab, n = nd_label(mask.astype(bool))
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    return lab == counts.argmax()

def get_edge_values(img: np.ndarray, mask: np.ndarray, axis: str):
    assert axis in ("x", "y")
    h, w = img.shape[:2]
    if axis == "y":
        rows = np.where(mask.any(axis=1))[0]
        y_top = int(rows.min()) if rows.size else 0
        y_bottom = int(rows.max()+1) if rows.size else h
        return y_top, y_bottom
    else:
        cols = np.where(mask.any(axis=0))[0]
        x_left = int(cols.min()) if cols.size else 0
        x_right = int(cols.max()+1) if cols.size else w
        return x_left, x_right

def get_bottommost_pixels(img: np.ndarray, mask: np.ndarray, y_edge_bottom: int):
    if y_edge_bottom <= 0:
        return 0, np.array([], dtype=int)
    y = int(y_edge_bottom - 1)
    xs = np.where(mask[y])[0]
    return y, xs

def include_buffer_y_axis(img: np.ndarray, y_top: int, y_bottom: int, buffer_size: int):
    h = img.shape[0]
    y_top = max(0, int(y_top - buffer_size))
    y_bottom = min(h, int(y_bottom + buffer_size))
    if y_top >= y_bottom:
        y_top, y_bottom = 0, h
    return y_top, y_bottom

def get_distance_from_starting_side(img: np.ndarray, mode: str, x_left: int, x_right: int):
    mask = (img > 0)
    col_sum = mask.sum(axis=0)
    nz = np.where(col_sum > 0)[0]
    if nz.size == 0:
        return 0
    if mode == "left":
        return int(nz[0])
    else:
        w = img.shape[1]
        return int((w - 1) - nz[-1])

def include_buffer_x_axis(img: np.ndarray, mode: str, x_left: int, x_right: int, buffer_size: int):
    w = img.shape[1]
    x_left = max(0, int(x_left - buffer_size))
    x_right = min(w, int(x_right + buffer_size))
    if x_left >= x_right:
        x_left, x_right = 0, w
    return x_left, x_right

def convert_bottommost_pixels_wrt_cropped_image(mode: str, bottom_y: int, bottom_xs: np.ndarray,
                                                 y_edge_top: int, x_edge_right: int, x_edge_left: int):
    h_c = None  # not needed here
    w_c = int(x_edge_right - x_edge_left)
    y_c = int(bottom_y - y_edge_top)
    xs_c = (bottom_xs - x_edge_left).astype(int)
    xs_c = xs_c[(xs_c >= 0) & (xs_c < w_c)]
    if mode == "right" and w_c > 0:
        xs_c = (w_c - 1) - xs_c
    return y_c, xs_c

def get_rightmost_pixels_wrt_cropped_image(mode: str, cropped_mask: np.ndarray, find_rightmost_from_ratio: float):
    h, w = cropped_mask.shape[:2]
    start_y = int(max(0, min(h - 1, np.floor(h * (1.0 - float(find_rightmost_from_ratio))))))
    m = cropped_mask if mode == "left" else np.fliplr(cropped_mask)
    ys = []
    xs_right = []
    for y in range(start_y, h):
        if m[y].any():
            ys.append(y)
            xs_right.append(np.max(np.where(m[y])[0]))
    if len(ys) == 0:
        return np.array([start_y]), 0
    y_start, y_end = int(min(ys)), int(max(ys))
    x_right = int(max(xs_right))
    return np.array([y_start, y_end]), x_right

# Required API
def crop_img_from_largest_connected(img, mode, erode_dialate=True, iterations=100,
                                    buffer_size=50, find_rightmost_from_ratio=1/3):
    """
    Performs erosion on the mask of the image, selects largest connected component,
    dilates it back, and computes a buffered crop and key points.

    Returns: (window_location, rightmost_points, bottommost_points, distance_from_starting_side)
    """
    assert mode in ("left", "right")
    assert img.ndim == 2, "img must be 2D grayscale"
    img_mask = img > 0
    if erode_dialate and iterations > 0:
        img_mask = binary_erosion(img_mask, iterations=iterations)
    largest_mask = get_mask_of_largest_connected_component(img_mask)
    if erode_dialate and iterations > 0:
        largest_mask = binary_dilation(largest_mask, iterations=iterations)

    y_edge_top, y_edge_bottom = get_edge_values(img, largest_mask, "y")
    x_edge_left, x_edge_right = get_edge_values(img, largest_mask, "x")

    bottommost_nonzero_y, bottommost_nonzero_x = get_bottommost_pixels(img, largest_mask, y_edge_bottom)
    y_edge_top, y_edge_bottom = include_buffer_y_axis(img, y_edge_top, y_edge_bottom, buffer_size)
    distance_from_starting_side = get_distance_from_starting_side(img, mode, x_edge_left, x_edge_right)
    x_edge_left, x_edge_right = include_buffer_x_axis(img, mode, x_edge_left, x_edge_right, buffer_size)

    # convert bottommost pixel locations w.r.t. cropped image. Flip if necessary.
    bottommost_nonzero_y, bottommost_nonzero_x = convert_bottommost_pixels_wrt_cropped_image(
        mode,
        bottommost_nonzero_y,
        bottommost_nonzero_x,
        y_edge_top,
        x_edge_right,
        x_edge_left
    )

    # rightmost from bottom portion (on cropped mask)
    cropped_mask = largest_mask[y_edge_top: y_edge_bottom, x_edge_left: x_edge_right]
    rightmost_nonzero_y, rightmost_nonzero_x = get_rightmost_pixels_wrt_cropped_image(
        mode, cropped_mask, find_rightmost_from_ratio
    )

    window_location = (y_edge_top, y_edge_bottom, x_edge_left, x_edge_right)
    rightmost_points = ((int(rightmost_nonzero_y[0]), int(rightmost_nonzero_y[-1])), int(rightmost_nonzero_x))
    bottommost_points = (int(bottommost_nonzero_y),
                         (int(bottommost_nonzero_x[0]) if bottommost_nonzero_x.size else 0,
                          int(bottommost_nonzero_x[-1]) if bottommost_nonzero_x.size else 0))
    return window_location, rightmost_points, bottommost_points, int(distance_from_starting_side)

def image_orientation(horizontal_flip, side):
    assert horizontal_flip in ['YES', 'NO'], "Wrong horizontal flip"
    assert side in ['L', 'R'], "Wrong side"
    if horizontal_flip == 'YES':
        return 'right' if side == 'R' else 'left'
    else:
        return 'left' if side == 'R' else 'right'

# Batch APIs (placeholders: require external project I/O utilities) 
def crop_mammogram(input_data_folder, exam_list_path, cropped_exam_list_path, output_data_folder,
                   num_processes, num_iterations, buffer_size):
    raise NotImplementedError("This function depends on project-specific I/O (pickling, data_handling, Pool).")

def crop_mammogram_one_image(scan, input_file_path, output_file_path, num_iterations, buffer_size):
    raise NotImplementedError("This function depends on reading_images/saving_images utilities.")

def crop_mammogram_one_image_short_path(scan, input_data_folder, output_data_folder,
                                        num_iterations, buffer_size):
    raise NotImplementedError("This function depends on project-specific path layout.")



def transformation_intensite(img, T, max_int=255):
    table = np.array([T(i) for i in range(max_int+1)]).clip(0, max_int).astype(np.uint8)
    img_transformee = cv2.LUT(img, table)  # Applique la fonction à chaque pixel
    return img_transformee



def min_max_scale_grayscale(grayscale_image):
    # convert to numpy array
    img_array = np.array(grayscale_image).astype(np.float32)
    # min-max scaling
    min_val = img_array.min()
    max_val = img_array.max()
    if max_val - min_val == 0:
        # avoid division by zero : return black image or same image
        scaled_array = np.zeros_like(img_array, dtype=np.uint8)
    else:
        scaled_array = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return scaled_array



def compute_mean_std(image_paths):
    n_pixels = 0
    channel_sum = 0.0
    channel_sum_squared = 0.0

    for img_path in tqdm(image_paths):
        img=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255.0  #Normalize to [0,1]
        #Add a channel dimension for consistency (H, W, C)
        img = np.expand_dims(img, axis=-1)
        #Accumulate sums
        n_pixels += (img.shape[0] * img.shape[1])
        channel_sum += np.sum(img, axis=(0, 1))
        channel_sum_squared += np.sum(img**2, axis=(0, 1))

    #Compute mean and std
    mean = channel_sum / n_pixels
    std = np.sqrt((channel_sum_squared / n_pixels) - (mean ** 2))
    return mean, std



#############################################################################################
#############################################################################################
#############################################################################################

def collect_png_paths(root_dir):
    """
    Explore the folders Normal, Benign, Cancer under `root_dir`,
    recursively, and collect paths of all .png files whose filenames
    do NOT contain 'Mask'. Return a pandas DataFrame with columns:
    ['path', 'class'].
    """

    classes = ["Normal", "Benign", "Cancer"]
    data = []

    # Ensure we only consider these three folders
    for cls in classes:
        class_dir = os.path.join(root_dir, cls)
        if not os.path.isdir(class_dir):
            print(f"Warning: directory not found -> {class_dir}")
            continue

        for dirpath, _, filenames in os.walk(class_dir):
            for fname in filenames:
                # Check file extension and exclude "Mask" in the name
                fname_lower = fname.lower()
                if fname_lower.endswith(".png") and "mask" not in fname_lower:
                    full_path = os.path.join(dirpath, fname)
                    data.append({
                        "path": full_path,
                        "class": cls.lower()  # normal / benign / cancer
                    })

    df = pd.DataFrame(data, columns=["path", "class"])
    return df



def crop_with_otsu_projection(
    img: np.ndarray,
    erode_iterations: int = 3,
    dilate_iterations: int = 5,
    high_intensity_cut: int = 245,
    max_saturation_frac: float = 0.9,
    proj_frac: float = 0.03,
    buffer_frac: float = 0.08,
) -> np.ndarray:
    """
    Robust breast crop:

      1) Otsu threshold to get foreground.
      2) Erosion + dilation to clean the mask.
      3) Remove mask-like components (very saturated / white).
      4) On the remaining mask:
           - compute row and column sums
           - keep rows/cols where sum > proj_frac * max_sum
           - add a small buffer
      5) Crop to that bounding box.

    This tends to keep the whole breast shape instead of a thin strip.
    """
    assert img.ndim == 2, "Expected a 2D grayscale image"
    h, w = img.shape

    # --- 1) Otsu mask ---
    t = threshold_otsu(img)
    mask = img > t

    # --- 2) Morphology ---
    if erode_iterations > 0:
        mask = binary_erosion(mask, iterations=erode_iterations)
    if dilate_iterations > 0:
        mask = binary_dilation(mask, iterations=dilate_iterations)

    if not mask.any():
        return img.copy()

    # --- 3) Remove white / mask-like components using saturation heuristic ---
    lab, n = nd_label(mask)
    if n == 0:
        return img.copy()

    saturated = img >= high_intensity_cut
    clean_mask = np.zeros_like(mask, dtype=bool)

    for lbl in range(1, n + 1):
        comp = (lab == lbl)
        area = comp.sum()
        if area == 0:
            continue
        sat_frac = saturated[comp].sum() / float(area)
        # keep components that are not almost fully saturated
        if sat_frac < max_saturation_frac:
            clean_mask |= comp

    if not clean_mask.any():
        # if we removed everything, just fall back to original mask
        clean_mask = mask

    # --- 4) Row/column projections to find tissue extent ---
    row_sum = clean_mask.sum(axis=1)  # shape (h,)
    col_sum = clean_mask.sum(axis=0)  # shape (w,)

    row_max = row_sum.max()
    col_max = col_sum.max()

    if row_max == 0 or col_max == 0:
        return img.copy()

    row_thresh = proj_frac * row_max
    col_thresh = proj_frac * col_max

    rows = np.where(row_sum > row_thresh)[0]
    cols = np.where(col_sum > col_thresh)[0]

    if rows.size == 0 or cols.size == 0:
        return img.copy()

    y_min, y_max = int(rows.min()), int(rows.max())
    x_min, x_max = int(cols.min()), int(cols.max())

    # --- 5) Add a buffer around the bounding box ---
    pad_y = int(buffer_frac * h)
    pad_x = int(buffer_frac * w)

    y_min = max(0, y_min - pad_y)
    y_max = min(h - 1, y_max + pad_y)
    x_min = max(0, x_min - pad_x)
    x_max = min(w - 1, x_max + pad_x)

    if y_max <= y_min or x_max <= x_min:
        return img.copy()

    return img[y_min:y_max+1, x_min:x_max+1]



def apply_clahe(img: np.ndarray,
                clip_limit: float = 2.0,
                tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    if img.dtype != np.uint8:
        img_min, img_max = float(img.min()), float(img.max())
        if img_max <= img_min:
            img_u8 = np.zeros_like(img, dtype=np.uint8)
        else:
            img_u8 = ((img - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
    else:
        img_u8 = img
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img_u8)


def zscore_normalize(img: np.ndarray) -> np.ndarray:
    img_f = img.astype(np.float32)
    mean = img_f.mean()
    std = img_f.std()
    if std < 1e-6:
        std = 1.0
    return (img_f - mean) / std



def preprocess_mammogram_from_path(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at: {path}")

    cropped = crop_with_otsu_projection(img)
    clahe_img = apply_clahe(cropped)
    preproc = zscore_normalize(clahe_img)
    return preproc


def preprocess_and_display_mammogram(path: str,
                                     figsize=(12,6),
                                     cmap="gray",
                                     title_preproc="Preprocessed (Otsu + projection + CLAHE + z-score)",
                                     title_orig="Original"):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at: {path}")

    cropped = crop_with_otsu_projection(img)
    clahe_img = apply_clahe(cropped)
    preproc = zscore_normalize(clahe_img)

    plt.figure(figsize=figsize)

    # original
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap=cmap)
    plt.title(title_orig)
    plt.axis("off")

    # preprocessed
    plt.subplot(1, 2, 2)
    plt.imshow(preproc, cmap=cmap)
    plt.title(title_preproc)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return preproc



def preprocess_mammogram_from_path(path: str) -> np.ndarray:
    """
    Single-image preprocessing pipeline:
      - read grayscale
      - Otsu + projection-based crop
      - CLAHE
      - z-score normalization

    Returns
    -------
    preproc : np.ndarray
        Float32 array (H, W), mean≈0, std≈1.
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at: {path}")

    cropped = crop_with_otsu_projection(img)
    clahe_img = apply_clahe(cropped)
    preproc = zscore_normalize(clahe_img)
    return preproc


def preprocess_all_mammos(df: pd.DataFrame,
                          output_root: str,
                          overwrite: bool = False) -> pd.DataFrame:
    """
    Run preprocessing on all mammograms listed in df and save them as PNGs.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
            - 'path'  : original image path (PNG, etc.)
            - 'class' : label (e.g. 'normal', 'benign', 'cancer')
    output_root : str
        Root directory where preprocessed PNGs will be stored.
        Files are saved as:
            output_root / <class> / <basename>_preproc.png
    overwrite : bool
        If False, existing preprocessed files are reused.

    Returns
    -------
    preprocessed_df : pd.DataFrame
        Columns:
            - 'original_path'
            - 'preprocessed_path'
            - 'class'
    """
    os.makedirs(output_root, exist_ok=True)

    original_paths = []
    preproc_paths = []
    classes = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        orig_path = row["path"]
        cls = row["class"]

        # class subfolder
        class_dir = os.path.join(output_root, str(cls))
        os.makedirs(class_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(orig_path))[0]
        out_path = os.path.join(class_dir, base_name + "_preproc.png")

        if (not overwrite) and os.path.exists(out_path):
            # reuse existing file
            original_paths.append(orig_path)
            preproc_paths.append(out_path)
            classes.append(cls)
            continue

        try:
            preproc_img = preprocess_mammogram_from_path(orig_path)  # float32, z-scored
        except Exception as e:
            print(f"[WARNING] Skipping {orig_path}: {e}")
            original_paths.append(orig_path)
            preproc_paths.append(None)
            classes.append(cls)
            continue

        # Convert to uint8 for PNG saving:
        # (we rescale per-image back to 0–255 for visualization/storage)
        if preproc_img.dtype != np.uint8:
            vmin = float(preproc_img.min())
            vmax = float(preproc_img.max())
            if vmax <= vmin:
                to_save_u8 = np.zeros_like(preproc_img, dtype=np.uint8)
            else:
                to_save_u8 = ((preproc_img - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
        else:
            to_save_u8 = preproc_img

        cv2.imwrite(out_path, to_save_u8)

        original_paths.append(orig_path)
        preproc_paths.append(out_path)
        classes.append(cls)

    preprocessed_df = pd.DataFrame({
        "original_path": original_paths,
        "preprocessed_path": preproc_paths,
        "class": classes,
    })

    return preprocessed_df



class MammogramDataset(Dataset):
    def __init__(self, df: pd.DataFrame, mode: str, mean: float, std: float):
        """
        df must contain:
            - 'preprocessed_path'
            - 'class'
        mode: 'train' or 'test' or 'val'
        mean/std: grayscale normalization stats
        """
        assert mode in ["train", "test", "val"], "mode must be train / test / val"

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.mean = float(mean)
        self.std = float(std)
        self.targets = df["class"].values

        # ---------- TRANSFORMS ----------
        if self.mode == "train":
            self.transform = transforms.Compose([
            transforms.Resize((1200, 700)),

            # --- geometric ---
            transforms.RandomRotation(degrees=15),                     # ±15°
            transforms.RandomResizedCrop((1200, 700),
                                        scale=(0.9, 1.0),             # small crop
                                        ratio=(0.95, 1.05)),          # tiny aspect jitter
            transforms.RandomHorizontalFlip(p=0.5),

            # --- intensity ---
            transforms.ColorJitter(
                brightness=0.1,    # ±10%
                contrast=0.1,      # ±10%
            ),

            # --- noise/blur ---
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            ], p=0.3),
            transforms.RandomApply([
                transforms.RandomAdjustSharpness(sharpness_factor=1.5)
            ], p=0.3),

            # convert + normalize
            transforms.ToTensor(),
            transforms.Normalize(mean=[self.mean], std=[self.std])
        ])
        else:  # test OR val
            self.transform = transforms.Compose([
                transforms.Resize((1200, 700)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[self.mean], std=[self.std])
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row["preprocessed_path"]
        label_str = row["class"]

        # Load grayscale PNG
        img = Image.open(img_path).convert("L")

        # Apply deterministic or augmented transforms
        img = self.transform(img)

        # Convert class name to integer
        label = torch.tensor(int(row["class"]), dtype=torch.long)

        return img, label
    


def build_resnet18_grayscale(num_classes: int = 3) -> nn.Module:

    # New torchvision API (PyTorch 2.x)
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    # ---- 1) Change first conv to accept 1 channel instead of 3 ----
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )

    # Initialize new conv weights by averaging the RGB weights
    with torch.no_grad():
        model.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

    # ---- 2) Change final fully-connected layer to output num_classes logits ----
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model



def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute top-1 accuracy for classification.
    """
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.size(0)
    return correct / total



def train_one_epoch(model, dataloader, optimizer, criterion, device, scaler, use_amp=True):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    n_samples = 0
    print('training...')
    for images, labels in tqdm(dataloader,total=len(dataloader)):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # AMP backward
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy_from_logits(outputs.detach(), labels) * batch_size
        n_samples += batch_size

    epoch_loss = running_loss / n_samples
    epoch_acc = running_acc / n_samples
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc = 0.0
    n_samples = 0
    print('validating...')
    for images, labels in tqdm(dataloader,total=len(dataloader)):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy_from_logits(outputs, labels) * batch_size
        n_samples += batch_size

    epoch_loss = running_loss / n_samples
    epoch_acc = running_acc / n_samples
    return epoch_loss, epoch_acc



def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs: int = 20,
    lr: float = 1e-3,
    device: str | torch.device | None = None,
    use_amp: bool = True,
    train_info_path: str | None = None,   # <-- new
    ckpts_path: str | None = None,        # <-- new
):
    """
    Train ResNet-18 model with:
        - Adam optimizer
        - Cosine annealing LR
        - AMP (mixed precision)
        - Checkpoint saving each epoch
        - History saving (pickle)

    Parameters
    ----------
    train_info_path : str
        Path to a .pkl file where the training history will be saved.
    ckpts_path : str
        Directory where checkpoints (state_dict) will be saved.

    Returns
    -------
    model, history
    """

    # ---------------- Setup ----------------
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler(enabled=use_amp and device.type == "cuda")

    # Prepare checkpoint directory
    if ckpts_path is not None:
        os.makedirs(ckpts_path, exist_ok=True)

    # Training history container
    history = []

    # ---------------- Training Loop ----------------
    for epoch in range(1, num_epochs + 1):
        print(f"Starting epoch {epoch}/{num_epochs}...")
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, use_amp
        )
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step()

        # ----- Save checkpoint -----
        ckpt_path = None
        if ckpts_path is not None:
            ckpt_path = os.path.join(ckpts_path, f"epoch_{epoch:03d}.pth")
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "lr": scheduler.get_last_lr()[0],
            }, ckpt_path)

        # ----- Save epoch info in memory -----
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": scheduler.get_last_lr()[0],
            "ckpt_path": ckpt_path,
        })

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"LR={scheduler.get_last_lr()[0]:.6f} | "
            f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"Val loss={val_loss:.4f} acc={val_acc:.4f}"
        )

        # ----- Save training info to pickle -----
        if train_info_path is not None:
            with open(train_info_path, "wb") as f:
                pickle.dump(history, f)

    return model, history


def custom_sampler(dataset):
    targets = torch.tensor(dataset.targets)
    class_counts = torch.bincount(targets)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[targets]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )