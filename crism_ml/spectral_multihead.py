import os
import argparse
from crism_ml.train import load_data
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy.io

# ---------------------------- Configuration ----------------------------
parser = argparse.ArgumentParser(description='CRISM Mineral Classifier Training')
parser.add_argument('--data_path', type=str, required=True, help='Path to .mat data file')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Input batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--hidden_size', type=int, default=128, help='LSTM hidden state size')
parser.add_argument('--num_classes', type=int, required=True, help='Number of mineral classes')
parser.add_argument('--num_bands', type=int, required=True, help='Number of spectral bands (will be padded/truncated)')
parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.makedirs(args.output_dir, exist_ok=True)
torch.manual_seed(42)
np.random.seed(42)

# ---------------------------- Data Loading & Processing ----------------------------
class CRISMDataset(Dataset):
    def __init__(self, spectra, labels, num_bands):
        """
        Args:
            spectra: List of spectral samples (variable length)
            labels: Corresponding mineral labels
            num_bands: Target number of bands (will pad/truncate to this)
        """
        self.spectra = []
        self.labels = []
        
        # Process each spectrum
        for spec, lbl in zip(spectra, labels):
            spec = spec.squeeze().astype(np.float32)
            
            # Pad or truncate to target length
            if len(spec) < num_bands:
                # Pad with zeros
                pad_width = num_bands - len(spec)
                spec = np.pad(spec, (0, pad_width), mode='constant')
            elif len(spec) > num_bands:
                # Truncate
                spec = spec[:num_bands]
                
            # Normalize
            spec = (spec - np.mean(spec)) / (np.std(spec) + 1e-8)
            self.spectra.append(spec)
            self.labels.append(lbl)
            
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        return torch.tensor(self.spectra[idx]), torch.tensor(self.labels[idx])

def load_matlab_data(file_path):
    """Load data from MATLAB .mat file"""
    mat = scipy.io.loadmat(file_path)
    spectra = mat['data']  # Adjust these keys based on your .mat structure
    labels = mat['labels'].squeeze()
    return spectra, labels

# ---------------------------- Model Architecture ----------------------------
class SpectralAttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=args.hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True
        )
        self.attention = nn.Sequential(
            nn.Linear(args.hidden_size*2, args.hidden_size),
            nn.Tanh(),
            nn.Linear(args.hidden_size, 1),
            nn.Softmax(dim=1)
        )
        self.fc = nn.Linear(args.hidden_size*2, args.num_classes)
        
    def forward(self, x):
        x = x.unsqueeze(-1)  # Add channel dimension
        lstm_out, _ = self.lstm(x)
        attention_weights = self.attention(lstm_out).squeeze()
        context = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        return self.fc(context), attention_weights


# ---------------------------- Training Plot Setup ----------------------------
def save_attention_maps(attention_weights, labels, epoch):
    """Save attention visualizations for each class"""
    # Average across attention heads and batch
    # attention_weights = attention_weights.mean(axis=(0,1))  # [seq_len]
    epoch_dir = os.path.join(args.output_dir, 'attention_maps', f'epoch_{epoch+1:03d}')
    os.makedirs(epoch_dir, exist_ok=True)
    # print(labels)
    unique_classes = np.unique(labels)
    print("UNIQUE CLASSES: " , unique_classes)
    for cls in unique_classes:
        cls_attention = attention_weights[labels == cls].mean(axis=0)
        plt.figure(figsize=(12, 6))
        plt.plot(range(args.num_bands), cls_attention)
        plt.title(f'Class {cls} Attention - Epoch {epoch+1}')
        plt.xlabel('Spectral Band')
        plt.ylabel('Attention Weight')
        plt.savefig(os.path.join(epoch_dir, f'class_{cls}_attention.jpg'))
        plt.close()

def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    """Save training metrics visualization"""
    plt.figure(figsize=(12, 6))
    print(train_losses, "\n","\n",val_losses,"\n","\n", train_accs,"\n","\n", val_accs)
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Classification Accuracy')
    plt.legend()
    
    plt.savefig(os.path.join(args.output_dir, 'training_metrics.jpg'))
    plt.close()

# ---------------------------- Training Setup ----------------------------
def main():
    # Load and prepare data
    spectra, labels, _ = load_data(args.data_path)
    dataset = CRISMDataset(spectra, labels, args.num_bands)
    
    # Split into train/val
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2)
    train_loader = DataLoader(
        torch.utils.data.Subset(dataset, train_idx),
        batch_size=args.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        torch.utils.data.Subset(dataset, val_idx),
        batch_size=args.batch_size
    )
    
    # Initialize model
    model = SpectralAttentionModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    # Training loop
    for epoch in range(args.epochs):
        print("#"*80)
        print("#"*20,f'Epoch {epoch+1}/{args.epochs}',"#"*20)
        print("#"*80)
        model.train()
        train_loss, train_correct = 0.0, 0
        
        for spectra, labels in train_loader:
            spectra = spectra.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(spectra)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_correct += (predicted == labels).sum().item()
            # total = labels.size(0)
            
        # Print training metrics
        print("-"*40)
        print(f'Training Loss: {train_loss/len(train_loader):.4f}')
        print(f'Training Accuracy: {train_correct/len(train_idx):.4f}')
        print("-"*40)
        # Validation
        model.eval()
        val_loss, val_correct = 0.0, 0
        all_attention = []
        all_labels = []
        with torch.no_grad():
            for spectra, labels in val_loader:
                spectra = spectra.to(device)
                labels = labels.to(device)
                
                outputs, attention = model(spectra)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                
                all_attention.append(attention.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        # Print metrics
        print(f'Epoch {epoch+1}/{args.epochs}')
        print(f'Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}')
        print(f'Train Acc: {train_correct/len(train_idx):.4f} | Val Acc: {val_correct/len(val_idx):.4f}')
        print('-'*50)
        # Save attention maps
        print("-"*20,"Saving attention maps...", "-"*20)
        save_attention_maps(
            attention_weights=np.concatenate(all_attention), 
            labels=np.concatenate(all_labels), 
            epoch=epoch)
        print("-"*30,"Plotting metrics...", "-"*30)
        train_losses.append(train_loss/len(train_loader)),
        val_losses.append(val_loss/len(val_loader)),
        train_accs.append(train_correct/len(train_idx)),
        val_accs.append(val_correct/len(val_idx))
        plot_metrics(
            train_losses, val_losses, 
            train_accs, val_accs
        )
        print("-"*20,"Saving model...", "-"*20)
        os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True) 
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_losses,
            'accuracy': val_accs,
        }, os.path.join(args.output_dir, 'checkpoints' ,f'model_epoch_{epoch+1}.pth'))
        print("#"*20,f"End of Epoch {epoch+1}/{args.epochs}", "#"*20)
        print("#"*80)
        
        

# if __name__ == '__main__':
#     main()