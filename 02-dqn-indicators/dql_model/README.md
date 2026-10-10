# dql_model/ (binaries intentionally NOT included)

The original folder `E:\Indicators\dql_model\` holds six files, all written 2025-09-23:

| file | size | what it is |
|---|---|---|
| bull_model.keras / bear_model.keras / sideways_model.keras | ~494 KB each | Keras Q-networks, one per market regime |
| bull_epsilon.pkl / bear_epsilon.pkl / sideways_epsilon.pkl | 21 B each | pickled float: that agent's current exploration rate (epsilon) |

**How they were produced:** by the DQN agents in this stage (`solpy_dql.py`, `live_sol_ddql.py`, `soldql.py`). Those scripts use `MODEL_DIR = "dql_model"`, build a small dense Q-network (e.g. Dense 256 → 128 → linear over actions in `soldql.py`), train it off an experience-replay buffer (`remember()` / batch replay, target `r + gamma * max Q(s')`, gamma ≈ 0.95) and decay epsilon each step (e.g. 0.3 → 0.05, decay 0.99995). They `model.save(...)` the network and pickle epsilon so a restart resumes learning where it left off rather than exploring again from scratch.

**Why three of them:** the idea was that a single Q-network fit to one market tends to learn that market's drift, so a separate agent was kept per regime (bull / bear / sideways) and the live script would pick whichever matched current conditions. The exact script that wrote these three regime files was not in the folders reviewed; the split is described from file names and the shared DQN code.

**Why not included:** they're opaque, can't be reviewed, were trained on a particular data window, and are useless or misleading without the exact feature pipeline and scaler. Retrain from code if ever needed.
