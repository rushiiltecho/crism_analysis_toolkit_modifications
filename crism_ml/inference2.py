import os
import argparse
from crism_ml.io import _generate_envi_header, load_image
from crism_ml.preprocessing import remove_continuum
import numpy as np
import torch
import json
from torch.nn import functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image
import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def load_crism_image_1(img_path):
    """Load a CRISM .img file and return as np array"""
    try:
        # pylint: disable=import-outside-toplevel
        from spectral.io import envi, spyfile

        band_select = np.r_[433:185:-1, 170:-1:68]

        fbase, _ = os.path.splitext(img_path)
        try:
            img = envi.open(f"{fbase}.hdr")
        except spyfile.FileNotFoundError:
            _generate_envi_header(f"{fbase}.lbl")
            img = envi.open(f"{fbase}.hdr")

        arr = img.load()
        return arr[:, :, band_select]
    except Exception as e:
        print(f"EXCEPTION: this occured while loading crism image:\n{e}")
        
class SpectralAttentionModel(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.attention = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Linear(hidden_size*2, num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(-1)  # Add channel dimension
        lstm_out, _ = self.lstm(x)
        attention_weights = self.attention(lstm_out).squeeze()
        context = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        return self.fc(context), attention_weights



def process_spectrum(spectrum, target_length=248):
    """Process a single spectrum to the target length"""
    if len(spectrum) < target_length:
        # Pad with last value
        padded = np.pad(spectrum, (0, target_length - len(spectrum)), mode='edge')
        return padded
    elif len(spectrum) > target_length:
        # Truncate to target length
        return spectrum[:target_length]
    else:
        return spectrum

def normalize_spectrum(spectrum):
    """Normalize a single spectrum"""
    mean = np.mean(spectrum)
    std = np.std(spectrum) + 1e-8
    return (spectrum - mean) / std

def create_rgb_from_hyperspectral(image_data):
    """Create an RGB representation from hyperspectral data"""
    bands = image_data.shape[2]
    
    # Select bands approximately corresponding to red, green, blue
    # These indices need to be adjusted based on your specific data's wavelengths
    if bands >= 200:
        r_idx = int(bands * 0.8)  # Red (~650-700nm)
        g_idx = int(bands * 0.5)  # Green (~510-570nm)
        b_idx = int(bands * 0.2)  # Blue (~450-490nm)
    else:
        r_idx = min(int(bands * 0.8), bands-1)
        g_idx = min(int(bands * 0.5), bands-1)
        b_idx = min(int(bands * 0.2), bands-1)
    
    # Create RGB image
    rgb = np.zeros((image_data.shape[0], image_data.shape[1], 3), dtype=np.float32)
    
    # Normalize each band to 0-1 range
    for i, idx in enumerate([r_idx, g_idx, b_idx]):
        band = image_data[:, :, idx].copy()
        if np.any(~np.isnan(band)):
            valid_mask = ~np.isnan(band)
            min_val = np.min(band[valid_mask])
            max_val = np.max(band[valid_mask])
            if max_val > min_val:
                band = (band - min_val) / (max_val - min_val)
            band[~valid_mask] = 0
        rgb[:, :, i] = band.squeeze()
    
    # Clip to 0-1 range and convert to uint8
    rgb = np.clip(rgb, 0, 1)
    rgb = (rgb * 255).astype(np.uint8)
    
    return rgb

def create_inference_batch(image_data, batch_size=128):
    """Create batches from image data for inference"""
    height, width, bands = image_data.shape
    
    # Reshape to [pixels, bands]
    pixels = image_data.reshape(-1, bands)
    
    # Create batches
    for i in range(0, len(pixels), batch_size):
        batch = pixels[i:i+batch_size]
        processed_batch = []
        valid_indices = []
        
        # Process each spectrum in the batch
        for j, spectrum in enumerate(batch):
            # Skip if all zeros or NaN
            if np.all(spectrum == 0) or np.any(np.isnan(spectrum)):
                continue
                
            processed = process_spectrum(spectrum, args.num_bands)
            normalized = normalize_spectrum(processed)
            processed_batch.append(normalized)
            valid_indices.append(i + j)
            
        if processed_batch:
            yield torch.tensor(np.array(processed_batch), dtype=torch.float32), valid_indices

def create_mineral_labels(num_classes, class_names=None):
    """Create labels for mineral classes"""
    if class_names is None:
        # Default class names if not provided
        class_names = [f"Mineral {i+1}" for i in range(num_classes)]
    
    # Make sure we have enough class names
    if len(class_names) < num_classes:
        for i in range(len(class_names), num_classes):
            class_names.append(f"Mineral {i+1}")
    
    return class_names[:num_classes]

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CRISM Mineral Classification Inference')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--img_path', type=str, required=True, help='Path to CRISM .img file')
    parser.add_argument('--output_dir', type=str, default='./inference_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference')
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM hidden state size (must match training)')
    parser.add_argument('--num_classes', type=int, required=True, help='Number of mineral classes')
    parser.add_argument('--num_bands', type=int, required=True, help='Number of spectral bands')
    parser.add_argument('--confidence_threshold', type=float, default=0.2, help='Prediction confidence threshold')
    parser.add_argument('--class_names', type=str, default=None, help='Comma-separated list of class names')
    parser.add_argument('--rgb_blend', type=float, default=0.7, help='Alpha blend value for RGB overlay (0-1)')
    
    global args
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Parse class names if provided
    class_names = None
    if args.class_names:
        class_names = args.class_names.split(',')
    mineral_labels = create_mineral_labels(args.num_classes, class_names)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the model
    print("Loading model...")
    model = SpectralAttentionModel(args.hidden_size, args.num_classes)
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # Check checkpoint structure
    print("\n======================= CHECKPOINT KEYS =======================\n", checkpoint.keys())
    
    # Load model weights - handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Try loading directly
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    # Load CRISM image
    print(f"Loading CRISM image from {args.img_path}...")
    try:
        image_data = load_crism_image_1(args.img_path)
        height, width, bands = image_data.shape
        print(f"Image loaded with dimensions: {height}x{width}x{bands}")
        
        # Remove continuum for every pixel in the image
        pixel_data = image_data.reshape(-1, bands)
        print(f"Removing continuum from image pixel data... \nASHAPE: {pixel_data.shape} ")
        image_data = remove_continuum(pixel_data[:, :228])[0].reshape(height, width, 228)
        print("Continuum removal complete.")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Generate RGB representation for visualization
    print("Creating RGB representation...")
    rgb_image = create_rgb_from_hyperspectral(image_data)
    
    # Initialize results
    print("Running inference on image...")
    class_predictions = {}
    classification_map = np.zeros((height, width), dtype=np.int32)
    confidence_map = np.zeros((height, width), dtype=np.float32)
    
    # Process image in batches
    with torch.no_grad():
        for batch, indices in create_inference_batch(image_data, args.batch_size):
            if len(batch) == 0:
                continue
                
            batch = batch.to(device)
            
            # Run inference
            outputs, _ = model(batch)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predictions = torch.max(probabilities, dim=1)
            
            # Process results
            for i, idx in enumerate(indices):
                y, x = idx // width, idx % width
                
                pred_class = predictions[i].item()
                conf_value = confidence[i].item()
                
                if conf_value >= args.confidence_threshold:
                    # Store pixel coordinates by class
                    if pred_class not in class_predictions:
                        class_predictions[pred_class] = []
                    
                    class_predictions[pred_class].append([int(x), int(y)])
                    
                    # Update classification map
                    classification_map[y, x] = pred_class + 1  # +1 so that 0 is background
                    confidence_map[y, x] = conf_value
    
    # Save results to JSON
    output_json = os.path.join(args.output_dir, 'class_predictions.json')
    print(f"Saving predictions to {output_json}...")
    
    # Prepare JSON output with class information
    json_output = {
        "metadata": {
            "image_path": args.img_path,
            "image_dimensions": [height, width, bands],
            "confidence_threshold": args.confidence_threshold
        },
        "class_info": {}
    }
    
    # Add class predictions with labels
    for class_id, coords in class_predictions.items():
        class_name = mineral_labels[class_id]
        json_output["class_info"][class_name] = {
            "class_id": int(class_id),
            "pixel_count": len(coords),
            "coordinates": coords
        }
    
    with open(output_json, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    # Create visualizations
    print("Creating visualizations...")
    # Exclude mineral number 2 from class predictions
    class_predictions = {k: v for k, v in class_predictions.items() if k != 1}  # Mineral 2 has index 1 (0-based)

    # Update visualizations to skip mineral number 2
    # 1. Create a colormap for classification
    num_classes_with_bg = args.num_classes + 1  # +1 for background
    colors = plt.cm.jet(np.linspace(0, 1, num_classes_with_bg))
    colors[0, 3] = 0  # Make background transparent
    cmap = ListedColormap(colors)

    # 2. Plot classification map
    plt.figure(figsize=(12, 10))
    classification_map_filtered = np.copy(classification_map)
    classification_map_filtered[classification_map_filtered == 2] = 0  # Set mineral 2 to background
    im = plt.imshow(classification_map_filtered, cmap=cmap, interpolation='nearest')
    plt.title('Mineral Classification Map (Excluding Mineral 2)')

    # Add colorbar with class labels
    filtered_labels = ['Background'] + [label for i, label in enumerate(mineral_labels) if i != 1]
    filtered_ticks = [i for i in range(num_classes_with_bg) if i != 2]  # Exclude tick for mineral 2

    cbar = plt.colorbar(im, ticks=filtered_ticks)  # Use filtered ticks
    cbar.set_ticklabels(filtered_labels)  # Set filtered labels

    # Save classification map
    plt.savefig(os.path.join(args.output_dir, 'classification_map_filtered.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Plot confidence map (no changes needed here)

    # 4. Create RGB overlay with classification
    plt.figure(figsize=(14, 12))

    # Normalize rgb_image using min-max scaling
    # rgb_norm = rgb_image.astype(np.float32)
    # min_val = np.min(rgb_norm)
    # max_val = np.max(rgb_norm)
    # if max_val > min_val:
    #     rgb_norm = (rgb_norm - min_val) / (max_val - min_val)

    # # Create initial visualization with RGB image
    plt.imshow(rgb_image)

    # Create RGBA overlay
    overlay = np.zeros((height, width, 4))  # Create new RGBA array
    for class_id in range(1, args.num_classes + 1):  # Skip background (0)
        if class_id == 2:  # Skip mineral 2
            continue
        mask = classification_map == class_id
        if np.any(mask):
            class_color = colors[class_id]
            overlay[mask] = [class_color[0], class_color[1], class_color[2], args.rgb_blend]

    # Add overlay on top of RGB image
    plt.imshow(overlay, interpolation='nearest')

    # Add legend
    legend_elements = []
    for i, label in enumerate(mineral_labels):
        if i == 1:  # Skip mineral 2
            continue
        legend_elements.append(
            Patch(facecolor=colors[i+1, :3], edgecolor='k', label=label)
        )

    plt.legend(handles=legend_elements, loc='upper right', 
               title='Mineral Classes', bbox_to_anchor=(1.1, 1))
    plt.title('Mineral Classification Overlay on RGB Image (Excluding Mineral 2)')
    plt.savefig(os.path.join(args.output_dir, 'classification_rgb_overlay_filtered.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Save class distribution statistics
    class_counts = {mineral_labels[k]: len(v) for k, v in class_predictions.items() if k != 1}  # Skip mineral 2
    plt.figure(figsize=(12, 8))
    plt.bar(class_counts.keys(), class_counts.values(), color=colors[1:args.num_classes+1, :3])
    plt.xticks(rotation=45, ha='right')
    plt.title('Mineral Class Distribution (Excluding Mineral 2)')
    plt.ylabel('Pixel Count')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'class_distribution_filtered.png'), dpi=300)
    plt.close()

    print(f"Inference complete. Processed {height}x{width} pixels.")
    total_classified = sum(len(coords) for coords in class_predictions.values())
    print(f"Found {total_classified} pixels ({total_classified/(height*width)*100:.2f}%) above confidence threshold.")
    print(f"Results saved to {args.output_dir}")

if __name__ == '__main__':
    main()