import sys
import time
import csv

sys.path.append(r"c:\REPO\Antigravity\AIPoker")
from core.evaluator import PokerEvaluator

def run_preflop_equity_sweep():
    evaluator = PokerEvaluator()
    ranks = '23456789TJQKA'
    
    # Generate 169 starting hands
    hand_strings = []
    # Pairs (13)
    for r in ranks:
        hand_strings.append(f"{r}{r}")
    # Suited and Offsuited (156)
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            r1, r2 = ranks[j], ranks[i]
            hand_strings.append(f"{r1}{r2}s")
            hand_strings.append(f"{r1}{r2}o")
            
    print(f"Generated {len(hand_strings)} starting hands. Beginning Monte Carlo simulation (10,000 runs per hand)...")
    
    results = []
    start_time = time.time()
    
    for idx, hand_str in enumerate(hand_strings):
        # Map to cards
        if len(hand_str) == 2:
            cards = [f"{hand_str[0]}s", f"{hand_str[1]}h"]
        elif hand_str[2] == 's':
            cards = [f"{hand_str[0]}s", f"{hand_str[1]}s"]
        else:
            cards = [f"{hand_str[0]}s", f"{hand_str[1]}h"]
            
        # Run MC simulation against 1 random hand
        equity, _ = evaluator.calculate_equity([], cards, num_opponents=1, num_simulations=10000)
        results.append((hand_str, equity))
        
        if (idx + 1) % 20 == 0:
            print(f"Processed {idx + 1}/{len(hand_strings)} hands... ({time.time() - start_time:.1f}s)")
            
    # Sort by equity descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Save as CSV in repo
    csv_path = r"c:\REPO\Antigravity\AIPoker\preflop_equities.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Hand", "Equity"])
        for hand, eq in results:
            writer.writerow([hand, f"{eq:.4f}"])
            
    # Save as Markdown table in repo
    md_path = r"c:\REPO\Antigravity\AIPoker\preflop_equities.md"
    with open(md_path, 'w') as f:
        f.write("# Preflop Starting Hand Equities (10k MC Runs vs 1 Random Hand)\n\n")
        f.write("| Rank | Hand | Equity |\n")
        f.write("| --- | --- | --- |\n")
        for rank, (hand, eq) in enumerate(results, 1):
            f.write(f"| {rank} | {hand} | {eq:.2%} |\n")
            
    # Copy markdown table to artifact directory
    artifact_path = r"C:\Users\zwonk\.gemini\antigravity-ide\brain\c68a647c-2540-4757-8bda-15d48c1088de\preflop_equities.md"
    with open(artifact_path, 'w') as f:
        f.write("# Preflop Starting Hand Equities (10k MC Runs vs 1 Random Hand)\n\n")
        f.write("| Rank | Hand | Equity |\n")
        f.write("| --- | --- | --- |\n")
        for rank, (hand, eq) in enumerate(results, 1):
            f.write(f"| {rank} | {hand} | {eq:.2%} |\n")
            
    print(f"\nCompleted in {time.time() - start_time:.2f} seconds!")
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved Markdown to: {md_path}")
    print(f"Saved Artifact to: {artifact_path}")

if __name__ == '__main__':
    run_preflop_equity_sweep()
