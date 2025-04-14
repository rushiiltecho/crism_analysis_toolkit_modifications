import numpy as np
import torch
import spectral
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import argparse
import os
import scipy.io

# ---------------------------- Configuration ----------------------------
parser = argparse.ArgumentParser(description='CRISM Mineral Classification Mapping')
parser.add_argument('--image_path', type=str, required=False, help='Path to CRISM image (.img)')
parser.add_argument('--hdr_path', type=str, required=True, help='Path to image header file (.hdr)')
parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
parser.add_argument('--num_classes', type=int, required=True, help='Number of mineral classes')
parser.add_argument('--num_bands', type=int, required=True, help='Number of spectral bands used in training')
parser.add_argument('--output_dir', type=str, required=True, help='Output directory for classification maps')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.output_dir, exist_ok=True)

# ---------------------------- Model Architecture (must match training) ----------------------------
class SpectralAttentionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_size=1,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 1),
            torch.nn.Softmax(dim=1)
        )
        self.fc = torch.nn.Linear(256, args.num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)
        attention_weights = self.attention(lstm_out).squeeze()
        context = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        return self.fc(context), attention_weights

# ---------------------------- Data Preprocessing ----------------------------
class CRISMInferenceDataset(Dataset):
    def __init__(self, spectra, num_bands):
        self.spectra = []
        self.original_data = spectra
        self.original_shape = spectra.shape
        
        # Flatten image to 2D array (pixels x bands)
        spectra_flat = spectra.reshape(-1, spectra.shape[-1])
        
        for spec in spectra_flat:
            spec = spec.squeeze().astype(np.float32)
            
            # Handle NaN/invalid values
            spec = np.nan_to_num(spec)
            
            # Skip zero-spectra (background)
            if np.all(spec == 0):
                self.spectra.append(None)
                continue
                
            # Pad/truncate and normalize
            if len(spec) < num_bands:
                pad_width = num_bands - len(spec)
                spec = np.pad(spec, (0, pad_width), mode='constant')
            elif len(spec) > num_bands:
                spec = spec[:num_bands]
            
            spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-8)
            self.spectra.append(spec)
            
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        if self.spectra[idx] is None:
            return torch.zeros(args.num_bands), False  # Invalid pixel marker
        return torch.tensor(self.spectra[idx]), True

# ---------------------------- Classification Functions ----------------------------
def load_crism_image():
    """Load ENVI format CRISM image"""
    img = spectral.open_image(args.hdr_path)
    arr = img.load()
    return arr

def create_class_maps(predictions, dataset, class_names):
    """Create classification maps for each mineral class"""
    # Reshape predictions to original image dimensions
    class_map = predictions.reshape(dataset.original_shape[:-1])
    
    # Create RGB base image using actual data (not shape tuple)
    rgb_img = np.stack([
        np.squeeze(dataset.original_data[..., 90]),  # R band
        np.squeeze(dataset.original_data[..., 50]),  # G band
        np.squeeze(dataset.original_data[..., 20])   # B band
    ], axis=-1)
    
    # Normalize RGB values
    rgb_img = (rgb_img - np.nanpercentile(rgb_img, 2)) / \
             (np.nanpercentile(rgb_img, 98) - np.nanpercentile(rgb_img, 2))
    rgb_img = np.clip(rgb_img, 0, 1)

    # Create maps for each class
    for class_id in range(args.num_classes):
        plt.figure(figsize=(12, 12))
        
        # Create mask for current class
        mask = (class_map == class_id).astype(float)
        
        # Overlay mask on RGB image
        plt.imshow(rgb_img)
        plt.imshow(mask, alpha=0.5, cmap='jet', vmin=0, vmax=1)
        plt.title(f'Mineral Class {class_id} - {class_names[class_id]}')
        plt.axis('off')
        
        # Save individual map
        plt.savefig(os.path.join(args.output_dir, f'class_{class_id}_map.jpg'),
                    bbox_inches='tight', pad_inches=0)
        plt.close()
# ---------------------------- Main Inference ----------------------------
def main():
    # Load model
    model = SpectralAttentionModel().to(device)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load CRISM image
    img = load_crism_image()
    dataset = CRISMInferenceDataset(img, args.num_bands)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # Process image in batches
    predictions = np.zeros(len(dataset), dtype=np.uint8)
    
    with torch.no_grad():
        for batch_idx, (batch_data, valid_mask) in enumerate(loader):
            valid_pixels = batch_data[valid_mask].to(device)
            
            if len(valid_pixels) == 0:
                continue
                
            outputs, _ = model(valid_pixels)
            batch_preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            # Get indices of valid pixels in this batch
            start_idx = batch_idx * loader.batch_size
            end_idx = start_idx + len(batch_data)
            valid_indices = np.where(valid_mask.numpy())[0] + start_idx
            
            # Update predictions
            predictions[valid_indices] = batch_preds
    
    # Generate classification maps
    class_names = ['Class_{}'.format(i) for i in range(args.num_classes)]  # Replace with actual names
    create_class_maps(predictions, dataset, class_names)
    
    print(f"Classification maps saved to {args.output_dir}")

if __name__ == '__main__':
    main()