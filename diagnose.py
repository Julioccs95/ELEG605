"""
diagnose.py - mini test: is the local search the bottleneck?

Run from the folder holding toy_conj.py:   python diagnose.py

It answers four questions with numbers, no transformer involved:
  1. What do random graphs score?          (the baseline you start from)
  2. What is a star worth?                 (the trivial construction)
  3. Does the local search improve at all?  (mean/max gain per start)
  4. Where do the searched graphs live?     (score distribution + spread)

If question 3 shows gain ~ 0, the move set is the bottleneck and nothing in
the model or the loop can fix it.
"""

import random
import numpy as np
import networkx as nx

import toy_conj

TRIALS = 750
random.seed(0)
np.random.seed(0)




def random_bits():
    return "".join(random.choice("01") for _ in range(toy_conj.M))


def star_bits():
    """Star: vertex 0 joined to all others. Uses the same edge order as toy_conj."""
    A = np.zeros((toy_conj.N, toy_conj.N), dtype=int)
    A[0, 1:] = 1
    A[1:, 0] = 1
    iu = np.triu_indices(toy_conj.N, k=1)
    return "".join(str(x) for x in A[iu])


def tree_bits():
    """Random tree, the seed type you switched to."""
    A = np.zeros((toy_conj.N, toy_conj.N), dtype=int)
    order = list(range(toy_conj.N))
    random.shuffle(order)
    for k in range(1, toy_conj.N):
        u, v = order[k], random.choice(order[:k])
        A[u, v] = A[v, u] = 1
    iu = np.triu_indices(toy_conj.N, k=1)
    return "".join(str(x) for x in A[iu])


def stats(scores, label):
    s = np.array(scores, dtype=float)
    print(f"{label:24s} n={len(s):4d} mean={s.mean():8.4f} std={s.std():7.4f} "
          f"min={s.min():8.4f} max={s.max():8.4f} "
          f"q25={np.percentile(s,25):8.4f} q75={np.percentile(s,75):8.4f}")


def main():
    print(f"N={toy_conj.N}  M={toy_conj.M}")
    print("-" * 100)

    # --- 1. baselines -------------------------------------------------------
    rand = [random_bits() for _ in range(TRIALS)]
    tous = [tree_bits() for _ in range(TRIALS)]
    stats([toy_conj.reward_calc(s) for s in rand], "random graphs")
    stats([toy_conj.reward_calc(s) for s in tous], "random trees")
    print(f"{'star':24s} score = {toy_conj.reward_calc(star_bits()):.4f}"
          "   <- trivial construction the search should beat")
    print("-" * 100)

    # --- 2. does the search improve? ---------------------------------------
    gains, before, after = [], [], []
    searched = []                      # the graph STRINGS after the search
    for s in rand:
        r = toy_conj.greedy_search_from_startpoint(None, s)[0]
        b, a = toy_conj.reward_calc(s), toy_conj.reward_calc(r)
        before.append(b); after.append(a); gains.append(a - b)
        searched.append(r)

    g = np.array(gains)
    print(f"search gain:            mean={g.mean():8.4f}  max={g.max():8.4f}  "
          f"improved {int((g > 1e-9).sum())}/{len(g)}")
    if abs(g.max()) < 1e-6:
        print("  >> gain is ~0: the move set is the bottleneck.")
        print("     Nothing in the model, the tokenizer or the loop can fix this.")
    else:
        print("  >> the search does move graphs; check how much it reaches.")
    print("-" * 100)

    # --- 3. where the searched graphs end up --------------------------------
    stats(before, "starts (random)")
    stats(after, "after local search")

    # how concentrated are the outputs? duplicates mean a collapsed attractor
    uniq = len(set(after))
    print(f"distinct searched graphs: {uniq}/{len(after)}")

    # coarse score buckets: near-identical graphs give near-identical scores
    buckets = {}
    for sc in after:
        buckets[round(sc, 1)] = buckets.get(round(sc, 1), 0) + 1
    top = sorted(buckets.items(), key=lambda kv: -kv[1])[:5]
    print("score buckets (0.1 wide), most common:", top)
    if len(top) <= 2 and top and top[0][1] > 0.6 * len(after):
        print("  >> one or two buckets hold most of the output: a single attractor.")
    print("-" * 100)

    # --- 4. connectivity of the searched graphs -----------------------------
    conn = sum(nx.is_connected(nx.from_numpy_array(toy_conj.bitstring_to_adj(s)))
               for s in searched)
    print(f"connected after search: {conn}/{len(after)}")

    # --- 5. the star is the plateau? ---------------------------------------
    star_s = toy_conj.reward_calc(star_bits())
    best_s = max(after)
    print(f"best searched {best_s:.4f} vs star {star_s:.4f} "
          f"-> {'search BEATS star' if best_s > star_s else 'search is BELOW star'}")

    print("-" * 100)
    print("counterexamples found (score > 0):", sum(1 for sc in after if sc > 1e-9))


if __name__ == "__main__":
    main()
