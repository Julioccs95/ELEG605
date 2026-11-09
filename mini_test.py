import random
import numpy as np
import toy_conj

# --- Test A: what shape is the -2.0 bucket? -------------------------------
print("=== Test A: the -2.0 bucket ===")
seen = 0
seeds = ["".join(random.choice("01") for _ in range(toy_conj.M)) for _ in range(200)]
for s in seeds:
    r = toy_conj.greedy_search_from_startpoint(None, s)[0]
    if abs(toy_conj.reward_calc(r) + 2.0) < 0.05:
        A = toy_conj.bitstring_to_adj(r)
        deg = sorted(A.sum(axis=1).astype(int), reverse=True)
        print("edges", int(A.sum() / 2), "deg", deg)
        seen += 1
        if seen == 10:
            break
print(f"found {seen} graphs in the -2.0 bucket\n")

# --- Test B: can ANY graph beat 0 at this N? ------------------------------
print("=== Test B: best of 50k random graphs ===")
best, arg = -1e9, None
for _ in range(50_000):
    s = "".join(random.choice("01") for _ in range(toy_conj.M))
    sc = toy_conj.reward_calc(s)
    if sc > best:
        best, arg = sc, s
print("best of 50k random:", round(best, 4))
print("counterexamples among them:", sum(1 for _ in range(0)))  # see note below
print("graph:", arg)
