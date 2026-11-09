import random
import toy_conj, vocab, model
import matplotlib.pyplot as plt #To make distribution graphs
import numpy as np
import networkx as nx


ROUNDS = 10
SEEDS  = 2000
KEEP   = 500
STEPS  = 300


def local_search_all(seeds):
    #Runs the greedy search for the bitstrings given and returns the improved score
    return [toy_conj.greedy_search_from_startpoint(None, s)[0] for s in seeds]

# round 0: random seeds -> local search -> keep the best
#Sample random 0 1 bitstrings M times. The join joins the individual bits into one string
#And do that for range(Seeds time)
graphs = ["".join(random.choice("01") for _ in range(toy_conj.M)) for _ in range(SEEDS)]



#first the local search is done in the graphs and sort them by the reward score
#reverse= True gives the descending order
#keep the KEEP amount designated the default is 50
#Set used to remove duplicates
db = sorted(set(local_search_all(graphs)), key=toy_conj.reward_calc, reverse=True)[:KEEP]

#Train the tokenizer
vocab.train_tokenizer(db, n_tokens=100)   # once, never retrain 

#Encode the string in db into token ids
#len(vocab.encode(s)) keeps track of how many token a graph uses
#max finds the largest token id
#10 extra positions are then added so the transformer then has room to add more tokens
max_len = max(len(vocab.encode(s)) for s in db) + 10
m = model.build_model(vocab.vocab_size(), max_len)   # build the model
w0 = m.lm_head.weight.detach().clone()          #This is to test the baseline weights to see if the model does update

for r in range(1, ROUNDS + 1):
    #This builds the list of encoded bitstrings from the db
    seqs = [vocab.encode(s) for s in db]
    #train the model
    model.train(m, seqs, max_len, steps=STEPS)
    #obtain the weights after the trained model
    w1 = m.lm_head.weight.detach()              
    print(f"round {r}: weight change = {(w1 - w0).abs().mean().item():.6f}")
    w0 = w1.clone()                              # the updated weight is now w0 to use to compare

    #Tell the transformer to generate new sequences which is then decoded from bpe id's into bitstring
    sampled = [vocab.decode(x) for x in model.sample(m, SEEDS, max_len)]
    #Put into local search algorithm
    improved = local_search_all(sampled)
    #concatenate both the db and improved and keep the top k results
    #set is added to remove duplicates
    db = sorted(set(db + improved), key=toy_conj.reward_calc, reverse=True)[:KEEP]

    scores = [toy_conj.reward_calc(s) for s in db]
    best_score = max(scores)

    plt.figure(figsize=(8,5))
    plt.hist(scores, bins=30, edgecolor="black")
    plt.axvline(best_score, color="red", linestyle="--",
                label=f"Best = {best_score:.4f}")

    plt.title(f"Score Distribution - Round {r}")
    plt.xlabel("Reward Score")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(f"score_distribution_round_{r}.png")
    plt.close()
   

    print(f"round {r}: best {toy_conj.reward_calc(db[0]):.4f} "
          f"| mean {sum(toy_conj.reward_calc(s) for s in db) / len(db):.4f} "
          f"| valid samples {sum(toy_conj.is_valid(s) for s in sampled)}/{len(sampled)} "
          f"| counterexamples {sum(toy_conj.reward_calc(s) > 0 for s in db)}")