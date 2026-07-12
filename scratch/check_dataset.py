import torch

dataset_dict = torch.load("tools/data/vectorized/nlh_combined_tensors.pt")
stage_action = dataset_dict['stage_action']

unique, counts = torch.unique(stage_action, return_counts=True)
dist = dict(zip(unique.tolist(), counts.tolist()))
print("Action counts in dataset (0: Fold, 1: Call, 2: Raise):")
print(dist)
