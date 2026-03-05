import torch
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from train import DRDataset, weak_transform, strong_transform
import sys
import os

# Define paths (relative to this script)
CSV_FILE = "../../train.csv"
DATA_DIR = "../../train_images"

def verify_balancing():
    print(f"Checking for data at: {os.path.abspath(DATA_DIR)}")
    if not os.path.exists(CSV_FILE) or not os.path.exists(DATA_DIR):
        print(f"Data files NOT found at {CSV_FILE} or {DATA_DIR}. Skipping full verification.")
        return

    print("Reading CSV...")
    df = pd.read_csv(CSV_FILE)
    
    print("Initializing Dataset...")
    # Initialize Dataset with both transforms as per updated train.py
    dataset = DRDataset(
        df=df, 
        root_dir=DATA_DIR, 
        weak_transform=weak_transform, 
        strong_transform=strong_transform
    )
    
    # Calculate Weights (Replicating logic from train.py)
    print("Calculating weights...")
    labels = df['diagnosis'].values
    
    class_counts = pd.Series(labels).value_counts().sort_index()
    print("Original Class Counts:", class_counts.to_dict())
    
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[label] for label in labels]
    sample_weights = torch.DoubleTensor(sample_weights)
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    loader = DataLoader(dataset, batch_size=16, sampler=sampler, num_workers=0)
    
    print("\n--- Verifying Batch Balance (Sampling 5 batches) ---")
    
    total_counts = {0:0, 1:0, 2:0, 3:0, 4:0}
    
    for i, (images, batch_labels) in enumerate(loader):
        if i >= 5: break
        
        counts = pd.Series(batch_labels.numpy()).value_counts().sort_index()
        print(f"Batch {i+1} Labels: {batch_labels.tolist()} -> Counts: {counts.to_dict()}")
        
        for label in batch_labels.numpy():
            total_counts[label] += 1
            
    print("\nTotal distribution defined in 5 batches (target: uniform):")
    print(total_counts)
    
    print("\nSuccessfully loaded batches. Transforms and balancing appear compatible.")

if __name__ == "__main__":
    try:
        verify_balancing()
    except Exception as e:
        print(f"Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
