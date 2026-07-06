import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

def main():
    print("[Training] Loading human vectorized dataset...")
    csv_path = os.path.join("data", "vectorized_human_hands.csv")
    if not os.path.exists(csv_path):
        print(f"[Error] Dataset '{csv_path}' not found. Run parse_human_data.py first.")
        return
        
    df = pd.read_csv(csv_path)
    print(f"[Training] Loaded {len(df)} samples.")
    print("[Training] Feature distribution:")
    print(df.describe().T[['mean', 'min', 'max']])
    
    # Cast label to integer before checking counts
    df["action"] = df["action"].astype(int)

    # Ensure all classes [0, 1, 2, 3, 4] have at least 5 samples by appending dummy rows
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

    # Re-cast label to integer after concatenation
    df["action"] = df["action"].astype(int)

    # Split features and labels
    X = df.drop(columns=["action"])
    y = df["action"]
    
    # Check class distribution
    print("\n[Training] Class distribution (0=FOLD, 1=CHECK, 2=CALL, 3=BET, 4=RAISE):")
    print(y.value_counts().sort_index())
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\n[Training] Training set size: {len(X_train)}")
    print(f"[Training] Testing set size:  {len(X_test)}")
    
    # Fit XGBoost Classifier
    print("\n[Training] Fitting XGBClassifier...")
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        eval_metric="mlogloss",
        num_class=5
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n[Training] Test Accuracy: {accuracy:.4f}")
    print("\n[Training] Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Save the binary model file
    binary_dir = os.path.join("core", "models", "binaries")
    os.makedirs(binary_dir, exist_ok=True)
    model_save_path = os.path.join(binary_dir, "xgboost_human.json")
    
    print(f"\n[Training] Saving model to: {model_save_path}...")
    model.save_model(model_save_path)
    print("[Training] Model saved successfully!")

if __name__ == '__main__':
    main()
