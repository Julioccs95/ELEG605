import numpy as np
import networkx as nx
import toy_conj
import matplotlib.pyplot as plt

#This is for conj 1
"""def explore_reward_calc(obj: str) -> float:
    '''
    This reward calc does not write into the counterexample txt file
    '''
    #Get rid of whitespace
    obj = obj.strip()
    if not toy_conj.is_valid(obj):
        return -1e6
    A = toy_conj.bitstring_to_adj(obj)
    G = nx.from_numpy_array(A)
    #If it is not connected, then add a panalty
    if not nx.is_connected(G):
        return -1000.0 - nx.number_connected_components(G)

    
    deg = A.sum(axis=1) #Calculates the degree values of the adjacency matrix
    Delta = float(deg.max()) #Calculates the max degree
    dbar = float(deg.mean()) #Calculates the average degree of the graph
    lam1 = float(np.linalg.eigvalsh(A)[-1]) #here we use eigenvalsh to compute symmetric matrices more efficiently
    score = 0.5 * (dbar + Delta) - lam1   # > 0 means conjecture violated
    return score """

#This is for non-trivial conjecture
def explore_reward_calc(obj: str) -> float:
    '''
    This reward calc does not write into the counterexample txt file
    '''
    #Get rid of whitespace
    obj = obj.strip()
    if not toy_conj.is_valid(obj):
        return -1e6
    A = toy_conj.bitstring_to_adj(obj)
    G = nx.from_numpy_array(A)
    #If it is not connected, then add a panalty
    if not nx.is_connected(G):
        return -1000.0 - nx.number_connected_components(G)

    
    mu = len(nx.max_weight_matching(G, maxcardinality=True))
    lower_bound = np.sqrt(A.shape[0] - 1) + 1
    lam1 = float(np.linalg.eigvalsh(A)[-1]) #here we use eigenvalsh to compute symmetric matrices more efficiently

    score = lower_bound - (lam1 + mu)
    return score 

def validate(bits: str):
    """Recompute the score and check the graph is a valid construction."""
    score =  explore_reward_calc(bits)
    if score == -1e6:
        return "Graph is not valid bit-string"
    if score <= -1000.0:
        return "Graph is disconnected"
    return score


# check every stored line
with open("Counterexamples.txt") as f:
    print("Starting validation...")
    #Enumerate(f,1) = start reading file f starting at index 1
    for i, line in enumerate(f, 1):
        #Strip the newline "\n" and split on tab "\t". Where \n is not shown but is silent
        stored_s, bits = line.rstrip("\n").split("\t")
        #Convert the stored score into a float
        stored = float(stored_s)
        #recalculate the reward functions
        recalc = validate(bits)
        #Check if the recalc is a str
        if isinstance(recalc, str):
            print(f"{i}: {recalc}")
        #returns if it not a counterexample
        elif recalc <= 1e-9:
            print(f"{i}: stored {stored:.6f} but recomputes to {recalc:.6f} — not a counterexample")
            #If they are not the same then state it
        elif abs(recalc - stored) > 1e-9:
            print(f"{i}: stored {stored:.6f} but recomputes to {recalc:.6f}")
        elif stored >= 0:
            A = toy_conj.bitstring_to_adj(bits)
            G = nx.from_numpy_array(A)
            nx.draw(G, with_labels=True)
            plt.show()
            break
    
    print("Validation complete.")