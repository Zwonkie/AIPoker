# AI Poker Backlog

- [ ] **Check Detection Algorithm**: Currently, our action sequence tracker inside `TableState` relies purely on financial stack differences (which captures bets, raises, and calls perfectly). Because checking exchanges no money, it is invisible to this diff logic. In the future, we need to implement a "Check Detector" by observing the visual `active_button` shifting from one player to the next while the `pot_size` remains static.
