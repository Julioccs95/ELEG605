import random
import toy_conj, vocab, model
import matplotlib.pyplot as plt #To make distribution graphs
import numpy as np
import networkx as nx
import random, torch

import time

def timed(tag, fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    print(f"[TIME] {tag}: {time.perf_counter() - t0:.2f}s")
    return out



"""# ---------------------------------------------------------------------------
# TEST SETTINGS (small sweep, not a paper-scale run)
# ---------------------------------------------------------------------------
RUN_TAG       = "test"      # goes in every filename so arms don't overwrite each other

ROUNDS  = 3        # 5 is fine for a real run; 3 is enough to see a trend in a test
SEEDS   = 200      # keep well above KEEP, so the pool exceeds the selection threshold
KEEP    = 50       # must be < unique bitstrings in the pool, or selection is a no-op
STEPS   = 300      # was 1500; the model still moves, the run stays quick

N_EIG      = 10
K_CLUSTERS = 10    # was 20; with KEEP=50, 20 clusters is too fine to balance

RANDOM_SEED = 0    # seeds random + the model, so the arms are comparable

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# --- the two comparison axes ---
SEED_SOURCE  = "random"   # "empty" | "random" | "tree" | "mixed"
SEARCH_MODE  = "two_flip" # "one_flip" | "two_flip"
SELECT_MODE  = "diverse"  # "diverse" | "plain"

PAIR_SAMPLES = 1000       # was 3000/2000; ~1/3 the per-step cost of a full run
"""







"""# ---------------------------------------------------------------------------
# FORMAL RUN
# ---------------------------------------------------------------------------
ROUNDS  = 7      # 5 in the paper-style run; 10 shows whether arms keep diverging
SEEDS   = 2000     # round-0 pool must be comfortably > KEEP after dedup
KEEP    = 500      # the paper's figure; keep the warning about pool size in mind
STEPS   = 1500     # your original; enough for the transformer to move each round

N_EIG      = 10
K_CLUSTERS = 10    # ~25 picks per cluster at KEEP=500

SEED_SOURCE  = "random"   # "empty" | "random" | "tree" | "mixed"
SEARCH_MODE  = "two_flip" # "one_flip" | "two_flip"
SELECT_MODE  = "diverse"  # "diverse" | "plain"
RANDOM_SEEDS = 42     # three seeds per arm, for spread

PAIR_SAMPLES = 3000"""

#Medium run
ROUNDS  = 7 #Pattern Boost rounds
SEEDS   = 1000      # Graph generations from sampler
KEEP    = 250      # How many graphs to keep
STEPS   = 750      # gradient steps per round to re-train the transformer

N_EIG      = 10 #Number of eigenvalues per graphs to keepfor diversity learning
K_CLUSTERS = 10    

SEED_SOURCE  = "empty" #sampling technique
SEARCH_MODE  = "two_flip" #Local-search
SELECT_MODE  = "plain" #toggle for diversity awareness
RANDOM_SEED  = 42

PAIR_SAMPLES = 1000   #how many random two-edge-flip pairs the local search tries at each stalled step before giving up and declaring a local optimum.




def local_search_all(seeds):
     #Runs the local search for the bitstrings given and returns the improved graphs
    return [toy_conj.greedy_search_from_startpoint(
        #pair_samples is the number of random two-edge-flip pairs the local search tries at each stalled step before it declares a local optimum.
                None, s, mode=SEARCH_MODE, pair_samples=PAIR_SAMPLES)[0]
            for s in seeds]

   
    


#CHANGED: remove duplicates up to graph isomorphism, not just identical strings.
#The same star has many different bitstrings (one per relabelling), which flooded the db.
def canon(s):
    G = nx.from_numpy_array(toy_conj.bitstring_to_adj(s))
    return nx.weisfeiler_lehman_graph_hash(G)


def dedup_top(strings, k):
    seen, out = set(), []
    for s in sorted(set(strings), key=toy_conj.reward_calc, reverse=True):
        h = canon(s)
        if h not in seen:
            seen.add(h)
            out.append(s)
        if len(out) == k:
            break
    return out


def random_bits():
    return "".join(random.choice("01") for _ in range(toy_conj.M))


# ---------------------------------------------------------------------------
#CHANGED: Diversify & Conquer selection (steps 1-2 of the plan, defined once)
# ---------------------------------------------------------------------------
_score_cache = {}
def score(s):
    # reward_calc is called many times per round, so cache it
    if s not in _score_cache:
        _score_cache[s] = toy_conj.reward_calc(s)
    return _score_cache[s]


def feature(s):
    """Step 1: isomorphism-invariant vector = top-N_EIG adjacency eigenvalues + edge density."""
    A = np.asarray(toy_conj.bitstring_to_adj(s), dtype=float)
    n = A.shape[0]
    eig = np.sort(np.linalg.eigvalsh(A))[::-1]      # eigenvalues only (relabelling-invariant)
    top = np.zeros(N_EIG)
    top[:min(N_EIG, n)] = eig[:N_EIG]               # pad with zeros if n < N_EIG
    #Density is the indicator to how full the graph is or the fraction of all possible edges that are actually present
    density = A.sum() / (n * (n - 1)) if n > 1 else 0.0
    return np.append(top, density)


def kmeans(X, k, iters=100, seed=0):
    """Plain NumPy k-means with k-means++ init. Returns a cluster label per row."""
    rng = np.random.default_rng(seed)
    n = len(X)
    k = min(k, n)
    centers = [X[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        if d2.sum() == 0:
            centers.append(X[rng.integers(n)])
        else:
            centers.append(X[rng.choice(n, p=d2 / d2.sum())])
    C = np.array(centers)

    labels = np.full(n, -1)
    for _ in range(iters):
        dist = ((X[:, None, :] - C[None]) ** 2).sum(-1)
        new = dist.argmin(1)
        if np.array_equal(new, labels):
            break                                   # assignments stable -> converged
        labels = new
        for j in range(k):
            members = X[labels == j]
            if len(members):
                C[j] = members.mean(0)
            else:                                   # empty cluster: re-seed on farthest point
                C[j] = X[dist.min(1).argmax()]
    return labels


def diverse_select(cands, keep, k):
    """Step 2: replaces the score-only db line. Returns bitstrings sorted best-first."""
    pool = sorted(set(cands))                       # step 4: exact-string dedup
    if len(pool) <= keep:
        print(f"  [diverse_select] pool {len(pool)} <= KEEP {keep}: no selection happening")
        return sorted(pool, key=score, reverse=True)

    X = np.array([feature(s) for s in pool])        # step 5: one feature vector per candidate
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)         # standardise so density isn't drowned out
    labels = kmeans(X, k)                           # step 6: k-means, nothing deleted

    clusters = {}
    for s, lab in zip(pool, labels):
        clusters.setdefault(lab, []).append(s)
    for lab in clusters:                            # step 7: best score first inside each cluster
        clusters[lab].sort(key=score, reverse=True)

    # visit clusters by their best member, so the global best is always picked first
    order = sorted(clusters, key=lambda l: score(clusters[l][0]), reverse=True)

    picked, idx = [], {l: 0 for l in order}         # step 8: round-robin until KEEP
    while len(picked) < keep:
        progressed = False
        for l in order:
            if idx[l] < len(clusters[l]):
                picked.append(clusters[l][idx[l]])
                idx[l] += 1
                progressed = True
                if len(picked) == keep:
                    break
        if not progressed:
            break

    sizes = sorted((len(v) for v in clusters.values()), reverse=True)
    print(f"  [diverse_select] pool {len(pool)} -> {len(picked)} | "
          f"{len(clusters)} clusters, sizes {sizes[:8]}...")

    # keep everything the diagnostics need
    _last_sel.update(pool=pool, X=X, labels=labels, clusters=clusters,
                     picked=picked, keep=keep, k=k)
    return sorted(picked, key=score, reverse=True)  # step 9: db[0] is still the best graph


# ---------------------------------------------------------------------------
#CHANGED: diagnostics for the diversity-aware selection
# ---------------------------------------------------------------------------
_last_sel = {}


def check_feature_invariance(s, trials=5):
    """CHECK 0: relabelling a graph must not change its feature vector."""
    A = np.asarray(toy_conj.bitstring_to_adj(s), dtype=float)
    n = A.shape[0]
    f0 = feature(s)
    worst = 0.0
    for _ in range(trials):
        p = np.random.permutation(n)
        Ap = A[p][:, p]
        eig = np.sort(np.linalg.eigvalsh(Ap))[::-1]
        top = np.zeros(N_EIG)
        top[:min(N_EIG, n)] = eig[:N_EIG]
        fp = np.append(top, Ap.sum() / (n * (n - 1)))
        worst = max(worst, np.abs(fp - f0).max())
    sym = np.allclose(A, A.T) and np.all(np.diag(A) == 0)
    print(f"[CHECK 0] adjacency symmetric, no self-loops: {sym} | "
          f"max feature change under relabelling: {worst:.2e} "
          f"({'OK' if worst < 1e-8 else 'NOT INVARIANT'})")


def diagnose(tag, prev_db=None):
    """Prints checks 1-5 for the most recent diverse_select call."""
    if not _last_sel:
        print(f"[{tag}] no clustering happened (pool <= KEEP), nothing to check")
        return
    pool, X, labels = _last_sel["pool"], _last_sel["X"], _last_sel["labels"]
    clusters, picked = _last_sel["clusters"], _last_sel["picked"]
    keep, k = _last_sel["keep"], _last_sel["k"]
    picked_set = set(picked)
    print(f"\n========== DIVERSITY DIAGNOSTICS: {tag} ==========")

    # CHECK 1: k-means itself
    C = np.array([X[labels == l].mean(0) for l in sorted(clusters)])
    inertia = sum(((X[labels == l] - C[i]) ** 2).sum()
                  for i, l in enumerate(sorted(clusters)))
    sizes = [len(clusters[l]) for l in sorted(clusters)]
    print(f"[CHECK 1] k-means: requested k={k}, non-empty clusters={len(clusters)}, "
          f"inertia={inertia:.1f}")
    print(f"          cluster sizes min/median/max = {min(sizes)}/"
          f"{int(np.median(sizes))}/{max(sizes)} "
          f"| singletons={sum(s == 1 for s in sizes)} "
          f"| largest cluster holds {max(sizes) / len(pool):.0%} of pool")

    # per-cluster profile: what shape each cluster is, and how many were picked
    print("          cluster | size | picked | best score | mean edges | top eigenvalue")
    for l in sorted(clusters, key=lambda l: score(clusters[l][0]), reverse=True):
        mem = clusters[l]
        edges = np.mean([np.asarray(toy_conj.bitstring_to_adj(s)).sum() / 2 for s in mem])
        lam1 = np.mean([feature(s)[0] for s in mem])
        n_pick = sum(s in picked_set for s in mem)
        print(f"          {l:7d} | {len(mem):4d} | {n_pick:6d} | {score(mem[0]):10.4f} "
              f"| {edges:10.1f} | {lam1:14.3f}")

    # CHECK 2: round-robin balance (the paper's cluster-balanced sampling)
    picks = [sum(s in picked_set for s in clusters[l]) for l in clusters]
    quota = keep / len(clusters)
    short = [l for l in clusters if len(clusters[l]) < quota]
    capped_ok = all(p == len(clusters[l]) for l, p in zip(clusters, picks)
                    if len(clusters[l]) < quota)
    print(f"[CHECK 2] round-robin: fair share ~{quota:.0f} per cluster | "
          f"picked per cluster min/max = {min(picks)}/{max(picks)} | "
          f"{len(short)} small clusters fully taken: {capped_ok}")
    # within each cluster the picked items must be the top-scoring ones
    order_ok = all(set(clusters[l][:p]) <= picked_set for l, p in zip(clusters, picks))
    print(f"          within-cluster picks are the top-scoring members: {order_ok}")

    # CHECK 3: unique shapes (WL hash = isomorphism classes)
    baseline = sorted(pool, key=score, reverse=True)[:keep]   # old score-only selection
    shapes = lambda ss: len({canon(s) for s in ss})
    print(f"[CHECK 3] unique shapes (WL hash): pool={shapes(pool)} | "
          f"diverse pick={shapes(picked)} | score-only top-{keep}={shapes(baseline)}")
    lab_of = dict(zip(pool, labels))
    print(f"          clusters covered: diverse={len({lab_of[s] for s in picked})} | "
          f"score-only={len({lab_of[s] for s in baseline})} of {len(clusters)}")

    # CHECK 4: quality is kept (paper: diversity without throwing away quality)
    ps = [score(s) for s in picked]
    bs = [score(s) for s in baseline]
    print(f"[CHECK 4] best score kept: {max(ps) == max(score(s) for s in pool)} | "
          f"mean diverse={np.mean(ps):.4f} vs score-only={np.mean(bs):.4f} | "
          f"overlap with score-only: {len(picked_set & set(baseline))}/{keep}")
    spread = lambda ss: np.mean(np.linalg.norm(
        X[[pool.index(s) for s in ss]] - X[[pool.index(s) for s in ss]].mean(0), axis=1))
    print(f"          feature spread (mean dist to centroid): diverse={spread(picked):.3f} "
          f"vs score-only={spread(baseline):.3f}  (diverse should be larger)")

    # CHECK 5: iterative refinement (selection is re-run each round on the new pool)
    if prev_db is not None:
        new = len(picked_set - set(prev_db))
        print(f"[CHECK 5] iterative refinement: {new}/{len(picked)} of db are new this round")
    print("=" * 52 + "\n")


# round 0: random seeds -> local search -> keep the best
# round 0 seed source: "empty" | "random" | "tree" | "mixed"
SEED_SOURCE = "empty"

if SEED_SOURCE == "empty":
    graphs = [toy_conj.empty_starting_point() for _ in range(SEEDS)]
elif SEED_SOURCE == "random":
    graphs = [toy_conj.random_graph_bits() for _ in range(SEEDS)]
elif SEED_SOURCE == "tree":
    graphs = ([toy_conj.random_tree_bits() for _ in range(SEEDS // 2)]
              + [toy_conj.random_double_star() for _ in range(SEEDS // 2)])
else:  # mixed
    graphs = ([toy_conj.random_graph_bits() for _ in range(SEEDS // 2)]
              + [toy_conj.random_tree_bits() for _ in range(SEEDS // 2)])

#CHANGED: initial database is also cluster-balanced

# round 0
t0 = time.perf_counter()
searched = local_search_all(graphs)
print(f"[TIME] round 0 local_search: {time.perf_counter() - t0:.1f}s "
      f"({len(graphs)} restarts)")

db = (diverse_select(searched, KEEP, K_CLUSTERS) if SELECT_MODE == "diverse"
      else sorted(set(searched), key=score, reverse=True)[:KEEP])

check_feature_invariance(db[0])
diagnose("round 0")


#Train the tokenizer
vocab.train_tokenizer(db, n_tokens=100)   # once, never retrain

max_len = max(len(vocab.encode(s)) for s in db) + 10
m = model.build_model(vocab.vocab_size(), max_len)   # build the model
w0 = m.lm_head.weight.detach().clone()

#baseline scores of random graphs, to compare against raw transformer samples
rand_scores = [toy_conj.reward_calc(random_bits()) for _ in range(SEEDS)]

for r in range(1, ROUNDS + 1):
    t_round = time.perf_counter()

    # step 10: train on db, sample, filter invalid, local search
    seqs = [vocab.encode(s) for s in db]

    t0 = time.perf_counter()
    model.train(m, seqs, max_len, steps=STEPS)
    t_train = time.perf_counter() - t0

    w1 = m.lm_head.weight.detach()
    print(f"round {r}: weight change = {(w1 - w0).abs().mean().item():.6f}")
    w0 = w1.clone()

    t0 = time.perf_counter()
    sampled = [vocab.decode(x) for x in model.sample(m, SEEDS, max_len, temperature=1.3)]
    t_sample = time.perf_counter() - t0

    #drop invalid samples instead of letting local search turn them into the empty graph
    n_valid = sum(toy_conj.is_valid(s) for s in sampled)
    valid = [s for s in sampled if toy_conj.is_valid(s)]

    t0 = time.perf_counter()
    improved = local_search_all(valid)                      # step 3
    t_search = time.perf_counter() - t0

    prev_db = db
    _last_sel.clear()

    # steps 4-9: selection, toggled by SELECT_MODE
    t0 = time.perf_counter()
    searched = db + improved
    db = (diverse_select(searched, KEEP, K_CLUSTERS) if SELECT_MODE == "diverse"
          else sorted(set(searched), key=score, reverse=True)[:KEEP])
    t_select = time.perf_counter() - t0

    print(f"[TIME] round {r}: train {t_train:.1f}s | sample {t_sample:.1f}s "
          f"| search {t_search:.1f}s | select {t_select:.1f}s "
          f"| total {time.perf_counter() - t_round:.1f}s")

    if SELECT_MODE == "diverse":
        diagnose(f"round {r}", prev_db)

    print("BESTGRAPH", db[0])

    scores = [score(s) for s in db]
    best_score = max(scores)

    # --- quick check: what do the best graphs look like? ---
    seen = {0, min(10, len(db) - 1), len(db) - 1}
    for j in sorted(seen):
        G = nx.from_numpy_array(toy_conj.bitstring_to_adj(db[j]))
        degs = sorted((d for _, d in G.degree()), reverse=True)
        print(f"round {r}: db[{j}] of {len(db)} degrees {degs[:5]} ... "
              f"| max {degs[0]} | edges {G.number_of_edges()}")

        # --- ADD THIS: draw the same graph ---
        pos = nx.spring_layout(G, seed=0)
        plt.figure(figsize=(6, 6))
        nx.draw_networkx(G, pos=pos, node_size=120, width=1.0,
                         with_labels=False, node_color="cornflowerblue")
        plt.title(f"round {r} — db[{j}] | score {score(db[j]):.4f} "
                  f"| edges {G.number_of_edges()} | max deg {degs[0]}")
        plt.axis("off")
        plt.savefig(f"bestgraph_round_{r}_db{j}.png", dpi=150)
        plt.close()


    #Plot 1: the kept db
    plt.figure(figsize=(8,5))
    plt.hist(scores, bins=30, edgecolor="black")
    plt.axvline(best_score, color="red", linestyle="--", label=f"Best = {best_score:.4f}")
    plt.title(f"Score Distribution - Round {r}")
    plt.xlabel("Reward Score")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"score_distribution_round_{r}.png")
    plt.close()

    #Plot 2: raw transformer samples (before local search) vs random graphs.
    #Disconnected graphs (score <= -1000) are left out so the histogram stays readable.
    raw = [score(s) for s in valid]
    raw_c = [x for x in raw if x > -1000]
    rand_c = [x for x in rand_scores if x > -1000]
    plt.figure(figsize=(8,5))
    if rand_c:
        plt.hist(rand_c, bins=30, alpha=0.5, label=f"random (connected {len(rand_c)})")
    if raw_c:
        plt.hist(raw_c, bins=30, alpha=0.5, label=f"transformer (connected {len(raw_c)})")
    plt.title(f"Raw samples vs random - Round {r}")
    plt.xlabel("Reward Score")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(f"raw_vs_random_round_{r}.png")
    plt.close()

    raw_mean = np.mean(raw_c) if raw_c else float("nan")
    raw_sd = np.std(raw_c, ddof=1) if len(raw_c) > 1 else float("nan")
    print(f"round {r}: best {best_score:.4f} "
          f"| mean {sum(scores) / len(scores):.4f} "
          f"| raw sample mean (connected) {raw_mean:.4f} sd {raw_sd:.4f} "
          f"| valid samples {n_valid}/{len(sampled)} "
          f"| counterexamples {sum(toy_conj.is_counterexample(s) for s in db)}")

    #save counterexamples here (reward_calc no longer writes files)
    with open(toy_conj.COUNTEREXAMPLE_FILE, "a") as f:
        for s in db:
            if toy_conj.is_counterexample(s):
                f.write(f"{r}\t{score(s)}\t{s}\n")
