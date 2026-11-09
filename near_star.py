import random, numpy as np, networkx as nx, toy_conj

best = -1e9
for i in range(20):
    s = toy_conj.near_star_starting_point(flips=random.randint(5, 25))
    r = toy_conj.greedy_search_from_startpoint(None, s)[0]
    sc = toy_conj.reward_calc(r)
    if sc > best:
        best = sc
        A = toy_conj.bitstring_to_adj(r)
        G = nx.from_numpy_array(A)
        print(f"[{i}] score={sc:+.4f} edges={int(A.sum()/2)} "
              f"mu={len(nx.max_weight_matching(G, maxcardinality=True))} "
              f"lam1={np.linalg.eigvalsh(A)[-1]:.3f}")
print("best:", f"{best:+.4f}")
