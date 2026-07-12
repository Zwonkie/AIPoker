import torch
dataset = torch.load("tools/data/vectorized/nlh_combined_tensors.pt")
context = dataset['context']
print("Context shape:", context.shape)
print("Min values:", context.min(dim=0).values)
print("Max values:", context.max(dim=0).values)
print("Mean values:", context.mean(dim=0).values)
