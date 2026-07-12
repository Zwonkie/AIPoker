# Model Output Processing & Action Selection

Once the neural network (Pluribus standalone model) computes the raw Expected Values (EV) for Folding, Calling, and Raising, the system maps these raw numbers into a probability distribution using a standard **Softmax** layer. A dice roll is then executed to select the final action.

This ensures the bot's decisions are probabilistic rather than deterministic, making its playstyle balanced and unexploitable.

## 1. Probability Calculation (Softmax)
To convert raw EV values (which can be any positive or negative numbers) into a valid probability distribution, the engine uses the Softmax function over the currently available actions. 

To prevent numerical overflow, the maximum raw EV is subtracted from each value before exponentiation (a standard mathematical optimization).

**Formula:**
$$P(a) = \frac{e^{EV(a) - \text{max\_EV}}}{\sum_{a'} e^{EV(a') - \text{max\_EV}}}$$

## 2. Action Selection (Dice Roll)
The engine generates a random number between `0.0` and `1.0` and selects the action based on the calculated Softmax probabilities.

## 3. Guardrails & Formatting
Before executing the chosen action, the system applies the following guardrails:
- **Free Checks**: If the chosen action is `FOLD` but checking is free (i.e., `call_amount = 0.0` and checking is available), the system overrides the decision to `CHECK`.
- **Check vs. Call**: If the chosen action is `CALL` and `call_amount = 0.0`, the action is renamed to `CHECK` for the GUI and action executor.

## Example Walkthrough
Let's say the engine outputs the following raw EVs:
* **Fold EV:** `0.37`
* **Call EV:** `3.76`
* **Raise EV:** `3.64`

Assuming all actions are available:
1. The maximum EV is `3.76`.
2. Compute the exponentiated differences:
   - Fold: $e^{0.37 - 3.76} = e^{-3.39} \approx 0.0337$
   - Call: $e^{3.76 - 3.76} = e^0 = 1.0$
   - Raise: $e^{3.64 - 3.76} = e^{-0.12} \approx 0.8869$
3. Sum of exponents = `1.9206`.
4. Calculate the probabilities:
   - **Fold Probability:** $0.0337 / 1.9206 \approx 1.8\%$
   - **Call Probability:** $1.0 / 1.9206 \approx 52.1\%$
   - **Raise Probability:** $0.8869 / 1.9206 \approx 46.1\%$
5. The bot rolls a random number (e.g. `0.45`). Since `0.45` falls within the Call range, it executes **CALL** (or **CHECK** if the call amount is 0).
