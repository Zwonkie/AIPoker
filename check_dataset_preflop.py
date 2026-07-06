import pandas as pd
# Load Pluribus
try:
    df_pro = pd.read_csv("data/vectorized_hands.csv")
    pre_pro_raise = df_pro[(df_pro['is_preflop'] == 1.0) & (df_pro['action'] == 4.0)]
    print("=== Pluribus Pre-flop Raise Samples ===")
    print(pre_pro_raise.head(10).to_string())
except Exception as e:
    print(e)
