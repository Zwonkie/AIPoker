import sys
import time
sys.path.append(r"c:\REPO\Antigravity\AIPoker")

from core.evaluator import PokerEvaluator

evaluator = PokerEvaluator()

start = time.time()
equity, reason = evaluator.calculate_equity([], ['As', 'Ah'], num_opponents=1, num_simulations=10000)
end = time.time()

print(f"Equity: {equity:.4f} ({reason})")
print(f"Time taken for 10,000 runs: {end - start:.2f} seconds")
