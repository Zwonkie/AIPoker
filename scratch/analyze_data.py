import torch
import collections

def analyze():
    print("Loading nlh_combined_tensors.pt...")
    data = torch.load("tools/data/vectorized/nlh_combined_tensors.pt")
    
    hole = data['hole'] # shape (N, 2)
    board = data['board'] # shape (N, 5)
    context = data['context'] # shape (N, 7)
    stage_action = data['stage_action'] # shape (N,)
    ev = data['ev'] # shape (N, 1)
    
    N = len(hole)
    print(f"Total training examples: {N}")
    
    # 1. Hole Cards Distribution
    # Flatten all hole cards (excluding 52 PAD)
    hole_flat = hole[hole != 52].tolist()
    hole_counts = collections.Counter(hole_flat)
    
    # Map index back to card string
    ranks = '23456789TJQKA'
    suits = 'cdhs'
    def int_to_card(val):
        if val == 52: return "PAD"
        s = val // 13
        r = val % 13
        return f"{ranks[r]}{suits[s]}"
        
    print("\n--- Hole Cards Frequencies ---")
    card_freqs = {int_to_card(k): v for k, v in hole_counts.items()}
    sorted_freqs = sorted(card_freqs.items(), key=lambda x: x[1], reverse=True)
    print("Top 5 most frequent hole cards:")
    for card, count in sorted_freqs[:5]:
        print(f"  {card}: {count} ({count/len(hole_flat)*100:.2f}%)")
    print("Bottom 5 least frequent hole cards:")
    for card, count in sorted_freqs[-5:]:
        print(f"  {card}: {count} ({count/len(hole_flat)*100:.2f}%)")
        
    # Standard deviation of card distribution to check uniformity
    all_counts = [hole_counts.get(i, 0) for i in range(52)]
    mean_count = sum(all_counts) / 52
    variance = sum((x - mean_count) ** 2 for x in all_counts) / 52
    std_dev = variance ** 0.5
    print(f"Uniformity metric: Mean card count = {mean_count:.1f}, Std Dev = {std_dev:.2f} ({std_dev/mean_count*100:.1f}% deviation)")
    
    # 2. Board Cards Distribution
    board_flat = board[board != 52].tolist()
    board_counts = collections.Counter(board_flat)
    board_freqs = {int_to_card(k): v for k, v in board_counts.items()}
    sorted_board_freqs = sorted(board_freqs.items(), key=lambda x: x[1], reverse=True)
    print("\n--- Board Cards Frequencies ---")
    print("Top 5 most frequent board cards:")
    for card, count in sorted_board_freqs[:5]:
        print(f"  {card}: {count} ({count/len(board_flat)*100:.2f}%)")
    print("Bottom 5 least frequent board cards:")
    for card, count in sorted_board_freqs[-5:]:
        print(f"  {card}: {count} ({count/len(board_flat)*100:.2f}%)")
        
    # 3. Context Features Analysis
    # Context columns: [Position, Bankroll, PotSize, Equity, PotOdds, NumOpponents, StreetLevel]
    features = ['Position', 'Bankroll', 'PotSize', 'Equity', 'PotOdds', 'NumOpponents', 'StreetLevel']
    print("\n--- Context Features Summary ---")
    for i, feat in enumerate(features):
        col = context[:, i]
        c_min = col.min().item()
        c_max = col.max().item()
        c_mean = col.mean().item()
        c_std = col.std().item()
        print(f"  {feat:15s} | Min: {c_min:6.2f} | Max: {c_max:6.2f} | Mean: {c_mean:6.2f} | StdDev: {c_std:6.2f}")
        
    # Check for anomalies: e.g. NaN, infinite values, or out of bound values
    nan_count = torch.isnan(context).sum().item()
    inf_count = torch.isinf(context).sum().item()
    print(f"\nContext anomalies: NaNs = {nan_count}, Infs = {inf_count}")

    # Generate Markdown Report Content
    report = []
    report.append("# Training Data Diagnostics & Analysis\n")
    report.append(f"Analyzed vectorized file `nlh_combined_tensors.pt` with **{N}** action-state examples.\n")
    
    report.append("## 1. Card Distribution Uniformity\n")
    report.append("In a standard randomly shuffled deck, cards should be roughly uniformly distributed. A high standard deviation or completely missing cards indicates a data collection hole.\n")
    
    report.append("### Hole Cards Uniformity")
    report.append(f"* **Mean Card Count**: {mean_count:.1f}")
    report.append(f"* **Standard Deviation**: {std_dev:.2f} ({std_dev/mean_count*100:.2f}% of mean)")
    report.append(f"* **Missing Cards**: {52 - len(hole_counts)} card(s) have 0 occurrences in the dataset.\n")
    
    report.append("| Card Rank/Suit | Top 5 Most Frequent | Count | % of Deck |")
    report.append("|---|---|---|---|")
    for card, count in sorted_freqs[:5]:
        report.append(f"| {card} | Yes | {count} | {count/len(hole_flat)*100:.2f}% |")
        
    report.append("\n| Card Rank/Suit | Bottom 5 Least Frequent | Count | % of Deck |")
    report.append("|---|---|---|---|")
    for card, count in sorted_freqs[-5:]:
        report.append(f"| {card} | Yes | {count} | {count/len(hole_flat)*100:.2f}% |")
        
    report.append("\n### Board Cards Uniformity")
    b_all_counts = [board_counts.get(i, 0) for i in range(52)]
    b_mean_count = sum(b_all_counts) / 52
    b_variance = sum((x - b_mean_count) ** 2 for x in b_all_counts) / 52
    b_std_dev = b_variance ** 0.5
    report.append(f"* **Mean Card Count**: {b_mean_count:.1f}")
    report.append(f"* **Standard Deviation**: {b_std_dev:.2f} ({b_std_dev/b_mean_count*100:.2f}% of mean)")
    report.append(f"* **Missing Cards**: {52 - len(board_counts)} card(s) have 0 occurrences in the dataset.\n")

    report.append("## 2. Context Vector Distribution")
    report.append("Checking for scaling bounds and ensuring data features cover the full range of poker situations.\n")
    report.append("| Feature | Min Value | Max Value | Mean Value | Std Dev | Description |")
    report.append("|---|---|---|---|---|---|")
    descriptions = {
        'Position': "Normalized position (0 to 0.9)",
        'Bankroll': "Hero stack size normalized by BB / 500",
        'PotSize': "Pot size normalized by BB / 500",
        'Equity': "Monte Carlo win equity (0% to 100%)",
        'PotOdds': "Pot odds facing a bet (0% to 100%)",
        'NumOpponents': "Number of opponents normalized by / 10",
        'StreetLevel': "Betting stage (0: Pre-flop, 0.33: Flop, 0.66: Turn, 1.0: River)"
    }
    for i, feat in enumerate(features):
        col = context[:, i]
        c_min = col.min().item()
        c_max = col.max().item()
        c_mean = col.mean().item()
        c_std = col.std().item()
        report.append(f"| {feat} | {c_min:.3f} | {c_max:.3f} | {c_mean:.3f} | {c_std:.3f} | {descriptions[feat]} |")
        
    report.append("\n## 3. Findings and Dataset Diagnostics")
    
    # Diagnostic checks
    findings = []
    if std_dev / mean_count > 0.15:
        findings.append("> [!WARNING]\n> **Hole Cards Distribution Skewed**: The deviation of card frequencies is quite high. Some cards are appearing much more often than others, indicating potential table selection or game recording biases in the source logs.")
    else:
        findings.append("> [!NOTE]\n> **Hole Cards Uniformity**: Hole cards are well-distributed with standard random variance. No structural holes detected.")
        
    if 52 - len(hole_counts) > 0:
        findings.append(f"> [!CAUTION]\n> **Missing Hole Cards**: {52 - len(hole_counts)} card(s) are completely missing from player hands in the dataset. This represents a model training blind spot.")
        
    if nan_count > 0 or inf_count > 0:
        findings.append(f"> [!CAUTION]\n> **Data Corruption Detected**: Found {nan_count} NaNs and {inf_count} Infs in the context features. This will destabilize training weights.")
    else:
        findings.append("> [!NOTE]\n> **Numerical Stability**: Context tensors contain 0 NaNs and 0 Infs, ensuring stable compilation.")
        
    report.extend(findings)
    
    # Save report
    import os
    os.makedirs('C:/Users/zwonk/.gemini/antigravity-ide/brain/c68a647c-2540-4757-8bda-15d48c1088de', exist_ok=True)
    with open('C:/Users/zwonk/.gemini/antigravity-ide/brain/c68a647c-2540-4757-8bda-15d48c1088de/training_data_analysis.md', 'w') as out_f:
        out_f.write("\n".join(report))
    print("\nSaved markdown report to C:/Users/zwonk/.gemini/antigravity-ide/brain/c68a647c-2540-4757-8bda-15d48c1088de/training_data_analysis.md")

if __name__ == '__main__':
    analyze()
