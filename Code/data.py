"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader

def get_loaders(data, data_path, batch_size, val_split=0.1):
    d_path = Path(data_path) / f"{data}_data.pt"
    data_dict = torch.load(d_path, weights_only=False)

    total_samples = data_dict['train_images'].shape[0]
    val_size = int(total_samples * val_split)
    val_start = total_samples - val_size

    # BUG FIX: was data_dict['train_images'] — included validation data in training
    # [:val_start] ensures training only uses first 90% — no overlap with validation
    train_data = data_dict['train_images'][:val_start]

    # BUG FIX: labels had shape [N,1] but CrossEntropyLoss needs shape [N]
    # .squeeze(1) removes the extra dimension — fixes the runtime crash
    train_labels = data_dict['train_labels'][:val_start].squeeze(1)

    val_data = data_dict['train_images'][val_start:]

    # BUG FIX: same label shape fix applied to validation labels
    val_labels = data_dict['train_labels'][val_start:].squeeze(1)

    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset   = TensorDataset(val_data, val_labels)

    # BUG FIX: same label shape fix applied to test labels
    test_dataset  = TensorDataset(data_dict['test_images'],data_dict['test_labels'].squeeze(1))

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(dataset=val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader