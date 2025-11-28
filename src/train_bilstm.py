# train_bilstm.py (ErgoSense - with Normalization + Scaler Saving)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

# Dataset and Preprocessing

class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

def create_sequences(df, seq_len=30):
    X, y = [], []
    values = df.drop('label', axis=1).values
    labels = df['label'].values
    for i in range(0, len(df) - seq_len):
        seq_x = values[i:i+seq_len]
        seq_y = labels[i+seq_len-1]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# BiLSTM Model Definition

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=2, dropout=0.3):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                              batch_first=True, bidirectional=True, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        out, _ = self.bilstm(x)
        emb = out.mean(dim=1)
        logits = self.classifier(emb)
        return logits, emb

def extract_embeddings(model, loader, device):
    model.eval()
    embs, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _, emb = model(x)
            embs.append(emb.cpu().numpy())
            labels.append(y.numpy())
    return np.concatenate(embs), np.concatenate(labels)


# Training Loop

def train_model(model, train_loader, val_loader, device, epochs=25, lr=1e-3):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'models/bilstm_model.pth')
    print(f" Best Validation Accuracy: {best_acc:.4f}")


# Main Training Flow

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)

    print(" Loading dataset...")
    df = pd.read_csv('dataset/posture_dataset.csv')

    # Drop unnecessary columns like timestamp
    feature_cols = [col for col in df.columns if col not in ['label', 'timestamp']]
    df = df[feature_cols + ['label']]
    print(f" Using {len(feature_cols)} feature columns.")

    # Normalize features (0-1 range)
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    joblib.dump(scaler, 'models/scaler.pkl')
    print(" Saved scaler to models/scaler.pkl")

    # Create sequences
    X, y = create_sequences(df, seq_len=30)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
    print(f" Train sequences: {X_train.shape}, Validation: {X_val.shape}")

    # Prepare Datasets
    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=64, shuffle=True)
    val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=64)

    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size = X.shape[2]
    model = BiLSTMEncoder(input_size=input_size)

    # Train model
    train_model(model, train_loader, val_loader, device)

    # Extract embeddings for SVM training
    print(" Extracting embeddings for SVM...")
    model.load_state_dict(torch.load('models/bilstm_model.pth'))
    embs_train, y_train = extract_embeddings(model, train_loader, device)
    embs_val, y_val = extract_embeddings(model, val_loader, device)

    np.savez('dataset/bilstm_embeddings.npz',
             X_train=embs_train, y_train=y_train,
             X_val=embs_val, y_val=y_val)

    print(" Saved BiLSTM embeddings for SVM at dataset/bilstm_embeddings.npz")
    print(" Training complete. Models and scaler are ready for inference.")
