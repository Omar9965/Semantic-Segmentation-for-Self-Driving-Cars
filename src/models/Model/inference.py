import torch
import os
import cv2
import numpy as np
from typing import List, Optional, Union, Dict, Any
import torch.nn.functional as F
from .unet import ResNet50UNet


# Paths
BEST_MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pth")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "output")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model instance (lazy loading)
_model: Optional[ResNet50UNet] = None

# Number of classes for semantic segmentation (Lyft-Udacity dataset)
NUM_CLASSES = 13

# Class names for self-driving car semantic segmentation
CLASS_NAMES = [
    "Background",       # 0
    "Building",         # 1
    "Fence",            # 2
    "Other",            # 3
    "Pedestrian",       # 4
    "Pole",             # 5
    "Road Line",        # 6
    "Road",             # 7
    "Sidewalk",         # 8
    "Vegetation",       # 9
    "Vehicle",          # 10
    "Wall",             # 11
    "Traffic Sign"      # 12
]

# Color map for visualization (RGB) - Based on CARLA/CityScapes palette for consistency
# These colors are designed to be high-contrast and visually distinct
CLASS_COLORS = np.array([
    [0, 0, 0],          # 0: Background - Black
    [70, 70, 70],       # 1: Building - Gray (CARLA standard)
    [190, 153, 153],    # 2: Fence - Light Gray/Pink (CARLA standard)
    [55, 90, 80],       # 3: Other - Teal (CARLA "Other")
    [220, 20, 60],      # 4: Pedestrian - Crimson Red (CARLA standard)
    [153, 153, 153],    # 5: Pole - Medium Gray (CARLA standard)
    [157, 234, 50],     # 6: Road Line - Bright Yellow-Green (CARLA RoadLine)
    [128, 64, 128],     # 7: Road - Purple (CARLA/CityScapes standard)
    [244, 35, 232],     # 8: Sidewalk - Magenta/Pink (CARLA standard)
    [107, 142, 35],     # 9: Vegetation - Olive Green (CARLA standard)
    [0, 0, 142],        # 10: Vehicle - Dark Blue (CARLA Car standard)
    [102, 102, 156],    # 11: Wall - Blue-Gray (CARLA standard)
    [220, 220, 0]       # 12: Traffic Sign - Yellow (CARLA standard)
], dtype=np.uint8)


def load_model(model_path: str = BEST_MODEL_PATH) -> ResNet50UNet:
    """Load the U-Net model with trained weights."""
    global _model
    if _model is None:
        _model = ResNet50UNet(n_classes=NUM_CLASSES, use_cbam=True)
        _model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        _model.to(DEVICE)
        _model.eval()
    return _model




def preprocess_image(image: np.ndarray) -> tuple[torch.Tensor, tuple[int, int]]:
    """Preprocess image for model inference."""
    # Ensure RGB format
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Store original size
    original_size = (image.shape[1], image.shape[0])
    
    # Resize to expected input size (256x256)
    image = cv2.resize(image, (256, 256))
    
    # Normalize with ImageNet mean/std (same as training)
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    
    # Convert to tensor [C, H, W]
    image = torch.from_numpy(image).permute(2, 0, 1).float()
    
    return image, original_size


def save_image(image: np.ndarray, filename: str) -> str:
    """Save image to output directory and return the URL path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(filepath, image)
    return f"/output/{filename}"


def create_colored_mask(mask: np.ndarray) -> np.ndarray:
    """Create a colored segmentation mask from class predictions."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx in range(NUM_CLASSES):
        colored[mask == class_idx] = CLASS_COLORS[class_idx]
    return colored


def create_overlay(original: np.ndarray, colored_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Create an overlay of the colored segmentation mask on the original image.
    
    Args:
        original: Original image in BGR format
        colored_mask: Colored segmentation mask in RGB format
        alpha: Blending factor (0.0-1.0), higher = more mask visibility
    
    Returns:
        Blended overlay image in BGR format
    """
    # Ensure mask is same size as original
    if colored_mask.shape[:2] != original.shape[:2]:
        colored_mask = cv2.resize(colored_mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Convert colored mask from RGB to BGR for proper blending with original (which is BGR)
    colored_mask_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
    
    # Blend
    overlay = cv2.addWeighted(original, 1 - alpha, colored_mask_bgr, alpha, 0)
    return overlay


def clean_segmentation_mask(mask: np.ndarray, kernel_size: int = 5, min_area: int = 100) -> np.ndarray:
    """
    Clean up segmentation mask to reduce noise and ensure dominant colors per object.
    
    Uses morphological operations and connected component analysis to:
    1. Remove small isolated noise pixels
    2. Fill small holes within objects
    3. Smooth object boundaries
    
    Args:
        mask: 2D numpy array with class IDs (0 to NUM_CLASSES-1)
        kernel_size: Size of morphological kernel (larger = more smoothing)
        min_area: Minimum area for a connected component to be kept
        
    Returns:
        Cleaned mask with same shape
    """
    cleaned_mask = mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Process each class separately (except background)
    for class_idx in range(1, NUM_CLASSES):
        # Create binary mask for this class
        class_binary = (mask == class_idx).astype(np.uint8) * 255
        
        if class_binary.sum() == 0:
            continue
        
        # Apply morphological closing (fill small holes)
        closed = cv2.morphologyEx(class_binary, cv2.MORPH_CLOSE, kernel)
        
        # Apply morphological opening (remove small noise)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        # Remove small connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
        
        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area < min_area:
                # Remove small components by setting them to 0
                opened[labels == label_idx] = 0
        
        # Update the cleaned mask where this class was cleaned
        # Only update pixels that were originally this class or are now filled
        cleaned_mask[opened > 0] = class_idx
    
    # Second pass: Apply median filter to smooth boundaries between classes
    # This helps with jagged edges between different classes
    cleaned_mask = cv2.medianBlur(cleaned_mask, 3)
    
    return cleaned_mask


def refine_mask_with_bilateral(mask: np.ndarray, original_image: np.ndarray) -> np.ndarray:
    """
    Refine segmentation mask using edge information from original image.
    Uses bilateral filtering approach to preserve edges while smoothing.
    
    Args:
        mask: 2D numpy array with class IDs
        original_image: Original BGR image for edge guidance
        
    Returns:
        Refined mask
    """
    # Convert original to grayscale for edge detection
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    
    # Detect edges
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges slightly
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges_dilated = cv2.dilate(edges, edge_kernel, iterations=1)
    
    # Create edge mask (where edges are NOT present)
    non_edge_mask = (edges_dilated == 0).astype(np.uint8)
    
    # Apply stronger smoothing only in non-edge regions
    smoothed = cv2.medianBlur(mask, 5)
    
    # Combine: keep original at edges, use smoothed elsewhere
    refined = np.where(non_edge_mask > 0, smoothed, mask)
    
    return refined.astype(np.uint8)


def segment_image(
    image: Union[str, np.ndarray],
    filename: str = "image",
    model: Optional[ResNet50UNet] = None,
    return_overlay: bool = True
) -> Dict[str, Any]:
    """
    Perform semantic segmentation on a single image.
    
    Args:
        image: Either a file path (str) or numpy array (BGR format)
        filename: Original filename (without extension) for output naming
        model: Optional pre-loaded model instance
        return_overlay: Whether to include overlay visualization
    
    Returns:
        Dict with URLs to saved images, dimensions, detected classes, and class counts
    """
    # Load model if not provided
    if model is None:
        model = load_model()
    
    # Load image if path is provided
    if isinstance(image, str):
        original_image = cv2.imread(image)
        if original_image is None:
            raise ValueError(f"Could not load image: {image}")
    else:
        original_image = image.copy()
    
    original_h, original_w = original_image.shape[:2]
    
    # Preprocess
    input_tensor, _ = preprocess_image(original_image)
    input_tensor = input_tensor.unsqueeze(0).to(DEVICE)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        # Multi-class: use argmax instead of sigmoid threshold
        pred_mask = torch.argmax(F.softmax(output, dim=1), dim=1)
    
    # Convert mask to numpy
    mask_np = pred_mask.squeeze().cpu().numpy().astype(np.uint8)
    
    # Resize mask back to original size (use INTER_NEAREST to preserve class labels)
    mask_resized = cv2.resize(mask_np, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    
    # Post-process mask to clean up noise and ensure dominant colors per object
    # Step 1: Clean segmentation with morphological operations
    mask_cleaned = clean_segmentation_mask(mask_resized, kernel_size=7, min_area=200)
    
    # Step 2: Refine mask using edge information from original image
    mask_refined = refine_mask_with_bilateral(mask_cleaned, original_image)
    
    # Create colored visualization mask (RGB format)
    colored_mask = create_colored_mask(mask_refined)
    
    # Convert colored mask to BGR for saving with cv2 (cv2.imwrite expects BGR)
    colored_mask_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
    
    # Generate unique suffix for this segmentation
    import time
    timestamp = int(time.time() * 1000)
    base_name = f"{filename}_{timestamp}"
    
    # Save images to output directory
    original_url = save_image(original_image, f"{base_name}_original.png")
    mask_url = save_image(colored_mask_bgr, f"{base_name}_mask.png")
    
    # Create and save overlay if requested
    overlay_url = None
    if return_overlay:
        overlay = create_overlay(original_image, colored_mask)
        overlay_url = save_image(overlay, f"{base_name}_overlay.png")
    
    return {
        "filename": filename,
        "original_image_url": original_url,
        "mask_url": mask_url,
        "overlay_url": overlay_url,
        "width": original_w,
        "height": original_h,
    }
    
def segment_multiple_images(
    images: List[tuple[Union[str, np.ndarray], str]],
    return_overlay: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Perform semantic segmentation on multiple images.
    
    Args:
        images: List of tuples (image_path_or_array, filename)
        return_overlay: Whether to include overlay visualizations
    
    Returns:
        Dict with 'results' list containing segmentation results
    """
    model = load_model()
    
    results = []
    for image, filename in images:
        result = segment_image(image, filename=filename, model=model, return_overlay=return_overlay)
        results.append(result)
    
    return {"results": results}


def get_class_names() -> List[str]:
    """Return the list of class names."""
    return CLASS_NAMES


def get_class_colors() -> np.ndarray:
    """Return the color map for visualization."""
    return CLASS_COLORS

