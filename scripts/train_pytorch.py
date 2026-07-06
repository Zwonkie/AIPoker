import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class PokerMLP(nn.Module):
    def __init__(self, input_dim=7, output_dim=5):
        super(PokerMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)

def main():
    print("[Training PyTorch] Loading vectorized dataset...")
    csv_path = os.path.join("data", "vectorized_hands.csv")
    if not os.path.exists(csv_path):
        print(f"[Error] Dataset '{csv_path}' not found. Run parse_data.py first.")
        return
        
    df = pd.read_csv(csv_path)
    df["action"] = df["action"].astype(int)

    # Balance classes if needed by adding dummy rows (same as xgboost script)
    dummy_rows = []
    for c in [0, 1, 2, 3, 4]:
        count = sum(df["action"] == c)
        if count < 5:
            for _ in range(5 - count):
                dummy_rows.append({
                    "is_preflop": 0.0,
                    "num_opponents": 1.0,
                    "equity": 0.5,
                    "pot_odds": 0.1,
                    "stack_pot_ratio": 10.0,
                    "bet_raise_available": 1.0,
                    "check_call_available": 1.0,
                    "action": float(c)
                })
    if dummy_rows:
        df = pd.concat([df, pd.DataFrame(dummy_rows)], ignore_index=True)
    df["action"] = df["action"].astype(int)

    X = df.drop(columns=["action"]).values.astype(np.float32)
    y = df["action"].values.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = PokerMLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Train
    model.train()
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    for epoch in range(150):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

    # Eval
    model.eval()
    with torch.no_grad():
        test_outputs = model(torch.tensor(X_test))
        preds = torch.argmax(test_outputs, dim=1).numpy()
        acc = np.mean(preds == y_test)
        print(f"[Training PyTorch] Test Accuracy: {acc:.4%}")

    # Export to ONNX
    binary_dir = os.path.join("core", "models", "binaries")
    os.makedirs(binary_dir, exist_ok=True)
    onnx_path = os.path.join(binary_dir, "poker_mlp.onnx")
    
    print(f"[Training PyTorch] Exporting model to ONNX: {onnx_path}")
    dummy_input = torch.randn(1, 7, dtype=torch.float32)
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        input_names=["features"], 
        output_names=["action_logits"],
        dynamic_axes={"features": {0: "batch_size"}, "action_logits": {0: "batch_size"}}
    )
    print("[Training PyTorch] Model exported successfully!")

if __name__ == '__main__':
    main()
