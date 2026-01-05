#!/usr/bin/env python3
"""Download AIME90 dataset from HuggingFace and save locally."""

from datasets import load_dataset

# Download from HuggingFace
print("Downloading AIME90 dataset from xiaoyuanliu/AIME90...")
dataset = load_dataset("xiaoyuanliu/AIME90")

# Save to local directory
print("Saving to ./datasets/AIME90...")
dataset.save_to_disk("./datasets/AIME90_hf")

print("Done! Dataset saved to ./datasets/AIME90_hf")
print(f"Number of samples: {len(dataset['train'])}")
print("\nSample data:")
print(dataset['train'][0])
