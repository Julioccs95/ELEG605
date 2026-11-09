#Packages of interest
#Note to self: Change to make a network x version of this code for simplicity
import numpy as np
import networkx as nx
import random

#Denote the number of vertices
N = 12
#The number of possible edges for the bitstring
M = int(N*(N-1) / 2)
COUNTEREXAMPLE_FILE = "Counterexamples.txt"


def empty_starting_point() -> str:
    ''' Create the empty starting string of all 0's for the graphs bitstring '''
    return "0" * M


def test_empty_starting_point():
    result = empty_starting_point()

    assert len(result) == M
    assert set(result) == {"0"}
    assert result == "0" * M

    print("empty_starting_point() passed")

test_empty_starting_point()




def bitstring_to_adj(s: str) -> np.ndarray:
    """
    This function is used to read off a bitstring and convert it into a adjacency matrix
    format
    """
    n = N
    #Create the empty adjacency matrix
    A = np.zeros((n, n), dtype=np.float64)
    #Index of the string value
    k = 0
    #The rows of the matrix
    for i in range(n):
        #The entries of the columns
        for j in range(i+1, n):
            #Check the value of the string bit
            if s[k] == "1":
                #If 1, then the adjacency matrix is symmetrically denoted as 1
                A[i, j] = 1.0
                A[j, i] = 1.0
            #Update k to move to 1 to move into the next bit-string
            k += 1
    #Return the Adjacency matrix
    return A

def bitstring_to_adj_v2(s: str) -> np.ndarray:
    "Same as above but it was made by claude as a more efficient alternative if needed"
    A = np.zeros((N, N))
    #triu gets the coordinated of the upper traingular matrix coordinates
    #k=1 means to omit the diagonal entris
    #returns this as cooridnates like (0,1), (0,2), (1,2)
    iu = np.triu_indices(N, k=1)  

    #(c == "1" for c in s) makes the string into true/false values
    #np.omiter() converts the values into numpy array 1.0 and 0.0
    #It just creates a numpy array from a iterator
    A[iu] = np.fromiter((c == "1" for c in s), dtype=float, count=M)
    return A + A.T


def is_valid(s:str) -> bool:
    #This checks if the length of the string is the exact match of the denoted M
    #This also checks if the string is made up of 0's and 1's.
    #Set coverts the string into it's unique characters
    return len(s) == M and set(s) <= {"0","1"}


#trivial reward_calc
"""def reward_calc(obj: str) -> float:
    #Get rid of whitespace
    obj = obj.strip()
    if not is_valid(obj):
        return -1e6
    A = bitstring_to_adj(obj)
    G = nx.from_numpy_array(A)
    #If it is not connected, then add a panalty
    if not nx.is_connected(G):
        return -1000.0 - nx.number_connected_components(G)

    
    deg = A.sum(axis=1) #Calculates the degree values of the adjacency matrix
    Delta = float(deg.max()) #Calculates the max degree
    dbar = float(deg.mean()) #Calculates the average degree of the graph
    lam1 = float(np.linalg.eigvalsh(A)[-1]) #here we use eigenvalsh to compute symmetric matrices more efficiently
    score = 0.5 * (dbar + Delta) - lam1   # > 0 means conjecture violated

    #In the same reward function, we write the counterexamples into a textfile
    
    if score > 1e-9:
        with open(COUNTEREXAMPLE_FILE, "a") as f:
            f.write(f"{score}\t{obj}\n")
    SEEN.add((score, obj)) if score > 1e-9 else None
    return score """

def neighbors_edge_flip(obj: str):
    '''
    May not need this perhaps
    '''
    # This function flips one bit of the bitstring, which will be used to assess a stronger local search
    s = list(obj)
    for k in range(len(s)):
        s[k] = "1" if s[k] == "0" else "0"
        #yield is like a return but continues on the entire loop generator
        yield "".join(s)
        #Flips back for the next value of the string
        s[k] = "1" if s[k] == "0" else "0"


#example on how to use it , but builds or flips each edge once
for neighbor in neighbors_edge_flip("001"):
    print(neighbor)

#This is v1
def greedy_search_from_startpoint(db, obj: str, max_steps=M):
    #Strip the whitespace first
    current = obj.strip()

    #Check if the object is valid, if not, then changes current to all 0 bit-string
    if not is_valid(current):
        current = empty_starting_point()

    #Calculates the current score
    current_score = reward_calc(current)

    #Start the loop for greedy search
    for _ in range(max_steps):
        #Conduct a list of the order index
        order = list(range(M))
        #This conducts the random shuffling of the positional index to avoid the bias 
        random.shuffle(order)
        #Improvement indicator
        improved = False
        for k in order:
            #Offers a candidate by cutting the string into halfs then concatenating by changing one bit at a time
            cand = current[:k] + ("1" if current[k] == "0" else "0") + current[k+1:]
            #Calculate the reward of the candidate 
            sc = reward_calc(cand)
            #If there is improvement, then we save the current object and the score and break the loop
            if sc > current_score:
                current, current_score = cand, sc
                improved = True
                break
        #If after the loop nothing is improved then we break the loop    
        if not improved:
            break
    #What is returned is the current object
    return [current]






#---------------------------------------------------------------------------------------------
#These are potential conjectures I may or may not use, individual components, etc etc

def find_matching_number(adj_matrix):
    graph = nx.Graph(adj_matrix)
    matching = nx.max_weight_matching(graph, maxcardinality=True)
    return len(matching)


def find_largest_eigenvalue(adj_matrix):
    adj_matrix = np.array(adj_matrix)
    eigenvalues = np.linalg.eigvals(adj_matrix)
    largest_eigenvalue = np.max(np.abs(eigenvalues))
    return largest_eigenvalue





'''

def reward_calc(obj: str) -> float:
    A = bitstring_to_adj(obj)

    if not np.allclose(A, A.T):
        return -999.0

    G = nx.from_numpy_array(A)

    if not nx.is_connected(G):
        return -999.0

    if any(G.has_edge(i, i) for i in G.nodes()):
        return -999.0

    lam1 = float(np.linalg.eigvalsh(A)[-1])
    mu = len(nx.max_weight_matching(G, maxcardinality=True))
    lower_bound = np.sqrt(A.shape[0] - 1) + 1

    reward =lower_bound - (lam1 + mu)
    return reward

'''




'''
def reward_calc(obj: str) -> float:
    A = bitstring_to_adj(obj)
    #Weighted conkjectire 2
    G = nx.Graph(A)
    m = G.number_of_nodes()
    if not nx.is_connected(G):
        return 15 * m
    eigenvalues = np.linalg.eigvals(A)
    largest_eigenvalue = np.max(eigenvalues)
    deg_values = dict(G.degree()).values()
    avg_deg = sum(deg_values) / G.number_of_nodes()
    max_degree = max([val for (node, val) in G.degree()])
    score = (0.9 * max_degree) + (0.1 * avg_deg)
    reward = largest_eigenvalue - score
    counterexample_found = 1 if score < largest_eigenvalue else 0
    if counterexample_found:
        print("ESTOPPPPPPPPPPPPPPPPPP")
    return reward

'''



'''
def reward_calc(obj: str) -> float:
    A = bitstring_to_adj(obj)
    G = nx.from_numpy_array(A)

    # Smooth connectivity penalty (instead of a flat -500)
    if not nx.is_connected(G):
        # penalize by how disconnected it is
        ncc = nx.number_connected_components(G)
        return -50.0 * (ncc - 1)

    # Real eigenvalues for symmetric matrix
    eigs = np.linalg.eigvalsh(A)
    lambda_max = float(eigs[-1])

    deg = np.array([d for _, d in G.degree()], dtype=float)
    avg_deg = float(deg.mean())
    max_deg = float(deg.max())

    # Your heuristic degree bound
    score = 0.9 * max_deg + 0.1 * avg_deg

    # Reward margin (positive is “counterexample-ish”)
    return lambda_max - score

'''



#non-tricvial conj
def reward_calc(obj: str) -> float:
    #Get rid of whitespace
    obj = obj.strip()
    if not is_valid(obj):
        return -1e6
    A = bitstring_to_adj(obj)
    G = nx.from_numpy_array(A)
    #If it is not connected, then add a panalty
    if not nx.is_connected(G):
        return -1000.0 - nx.number_connected_components(G)

    

    mu = len(nx.max_weight_matching(G, maxcardinality=True))
    lower_bound = np.sqrt(A.shape[0] - 1) + 1
    lam1 = float(np.linalg.eigvalsh(A)[-1]) #here we use eigenvalsh to compute symmetric matrices more efficiently

    score = lower_bound - (lam1 + mu)

    #In the same reward function, we write the counterexamples into a textfile
    """if score > 1e-9:
        with open(COUNTEREXAMPLE_FILE, "a") as f:
            f.write(f"{score}\t{obj}\n")"""
    return score



"""#Debug
def random_tree_bits():
    Random connected tree on N vertices, as a bitstring.
    A = np.zeros((N, N), dtype=int)
    order = list(range(N))
    random.shuffle(order)
    for k in range(1, N):
        u, v = order[k], random.choice(order[:k])
        A[u, v] = A[v, u] = 1
    iu = np.triu_indices(N, k=1)
    return "".join(str(x) for x in A[iu])"""






#This is v2
"""def greedy_search_from_startpoint(db, obj: str, max_steps=None):
    
    Plain score hill climb with four move types. Returns a one-element list.

    Accepts the first improving candidate in a randomly shuffled order, so two
    runs from the same seed can land in different places.
    
    if max_steps is None:
        max_steps = M

    current = obj.strip()
    if not is_valid(current):
        current = empty_starting_point()
    current_score = reward_calc(current)

    for _ in range(max_steps):
        improved = False

        # --- 1. single flip: delete or add one edge, random order -----------
        order = list(range(M))
        random.shuffle(order)
        for k in order:
            cand = _flip(current, k)
            if not is_valid(cand):
                continue
            sc = reward_calc(cand)
            if sc > current_score:
                current, current_score = cand, sc
                improved = True
                break
        if improved:
            continue

        # --- 2. swap: one edge off, another on (edge count stays fixed) -----
        for _try in range(200):
            s = list(current)
            ones = [i for i in range(M) if s[i] == "1"]
            zeros = [i for i in range(M) if s[i] == "0"]
            if not ones or not zeros:
                break
            s[random.choice(ones)] = "0"
            s[random.choice(zeros)] = "1"
            cand = "".join(s)
            if not is_valid(cand):
                continue
            sc = reward_calc(cand)
            if sc > current_score:
                current, current_score = cand, sc
                improved = True
                break
        if improved:
            continue

        # --- 3. add an edge (keeps connectivity for free) -------------------
        for _try in range(200):
            zeros = [i for i in range(M) if current[i] == "0"]
            if not zeros:
                break
            cand = _flip(current, random.choice(zeros))
            sc = reward_calc(cand)
            if sc > current_score:
                current, current_score = cand, sc
                improved = True
                break

        if not improved:               # genuine local optimum
            break

    return [current]"""