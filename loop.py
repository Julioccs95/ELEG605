import random
import toy_conj, vocab, model
import matplotlib.pyplot as plt #To make distribution graphs
import numpy as np
import networkx as nx


ROUNDS = 5
SEEDS  = 1000
KEEP   = 2000
STEPS  = 1500








def local_search_all(seeds):
    #Runs the greedy search for the bitstrings given and returns the improved score
    return [toy_conj.greedy_search_from_startpoint(None, s)[0] for s in seeds]


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


# round 0: random seeds -> local search -> keep the best
graphs = ([toy_conj.random_tree_bits() for _ in range(SEEDS // 2)]
          + [toy_conj.random_double_star() for _ in range(SEEDS // 2)])



db = sorted(set(local_search_all(graphs)), key=toy_conj.reward_calc, reverse=True)[:KEEP]



#Train the tokenizer
vocab.train_tokenizer(db, n_tokens=100)   # once, never retrain

max_len = max(len(vocab.encode(s)) for s in db) + 10
m = model.build_model(vocab.vocab_size(), max_len)   # build the model
w0 = m.lm_head.weight.detach().clone()

#CHANGED: baseline scores of random graphs, to compare against raw transformer samples
rand_scores = [toy_conj.reward_calc(random_bits()) for _ in range(SEEDS)]

for r in range(1, ROUNDS + 1):
    seqs = [vocab.encode(s) for s in db]
    model.train(m, seqs, max_len, steps=STEPS)
    w1 = m.lm_head.weight.detach()
    print(f"round {r}: weight change = {(w1 - w0).abs().mean().item():.6f}")
    w0 = w1.clone()

    #sampled = [vocab.decode(x) for x in model.sample(m, SEEDS, max_len)]

    sampled = [vocab.decode(x) for x in model.sample(m, SEEDS, max_len, temperature=1.3)]

    #CHANGED: drop invalid samples instead of letting local search turn them into the empty graph
    n_valid = sum(toy_conj.is_valid(s) for s in sampled)
    valid = [s for s in sampled if toy_conj.is_valid(s)]

    improved = local_search_all(valid)
    db = sorted(set(db + improved), key=toy_conj.reward_calc, reverse=True)[:KEEP]

    print("BESTGRAPH", db[0])          # <-- added here

    scores = [toy_conj.reward_calc(s) for s in db]
    best_score = max(scores)


    # --- quick check: what does the best graph look like? ---
        # --- quick check: what do the best graphs look like? ---
    seen = {0, min(10, len(db) - 1), len(db) - 1}
    for j in sorted(seen):
        degs = sorted((d for _, d in nx.from_numpy_array(
            toy_conj.bitstring_to_adj(db[j])).degree()), reverse=True)
        print(f"round {r}: db[{j}] of {len(db)} degrees {degs[:5]} ... "
              f"| max {degs[0]} | edges {sum(degs)//2}")



    #Plot 1 (as before): the kept db
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

    #CHANGED: Plot 2: raw transformer samples (before local search) vs random graphs.
    #This is where the PatternBoost distribution shift shows up.
    #Disconnected graphs (score <= -1000) are left out so the histogram stays readable.
    raw = [toy_conj.reward_calc(s) for s in valid]
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
    print(f"round {r}: best {best_score:.4f} "
          f"| mean {sum(scores) / len(scores):.4f} "
          f"| raw sample mean (connected) {raw_mean:.4f} "
          f"| valid samples {n_valid}/{len(sampled)} "
          f"| counterexamples {sum(toy_conj.is_counterexample(s) for s in db)}")  #CHANGED: uses tolerance

    #CHANGED: save counterexamples here (reward_calc no longer writes files)
    with open(toy_conj.COUNTEREXAMPLE_FILE, "a") as f:
        for s in db:
            if toy_conj.is_counterexample(s):
                f.write(f"{r}\t{toy_conj.reward_calc(s)}\t{s}\n")
