import random, numpy as np, networkx as nx, toy_conj

shown = 0
for _ in range(300):
    s = "".join(random.choice("01") for _ in range(toy_conj.M))
    r = toy_conj.greedy_search_from_startpoint(None, s)[0]
    sc = toy_conj.reward_calc(r)
    if sc > -3.05:
        A = toy_conj.bitstring_to_adj(r)
        G = nx.from_numpy_array(A)
        mu = len(nx.max_weight_matching(G, maxcardinality=True))
        print(f"score={sc:.3f} edges={int(A.sum()/2)} mu={mu} "
              f"lam1={np.linalg.eigvalsh(A)[-1]:.3f} deg={sorted(A.sum(axis=1).astype(int), reverse=True)}")
        shown += 1
        if shown == 8: break
